"""Use Channel Last and Cancel activation (gradient) checkpointing in the DiT training forward.

"""

from __future__ import annotations

import functools

import torch
from torch import nn

from .real_rope import rope_params_real


def compile_real_rope_freqs(original, options):
    """Rebuild self.freqs/freqs_action/freqs_state in the real cos/sin layout.

    """
    import os

    del options
    enable_tensorrt = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if enable_tensorrt:
            return
        d = self.dim // self.num_heads
        self.freqs_action = rope_params_real(1024 * 10, d)
        self.freqs_state = rope_params_real(1024, d)
        self.freqs = [
            rope_params_real(1024, d - 4 * (d // 6)),
            rope_params_real(1024, 2 * (d // 6)),
            rope_params_real(1024, 2 * (d // 6)),
        ]

    return __init__



def init_weights(self):
    r"""
    Initialize model parameters using Xavier initialization.
    """

    # basic init
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # init embeddings
    nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
    self.patch_embedding.to(memory_format=torch.channels_last_3d)
    for m in self.text_embedding.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
    for m in self.time_embedding.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)

    # init output layer
    nn.init.zeros_(self.head.head.weight)


def forward_train(
    self,
    x,
    timestep,
    timestep_action,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    y=None,
    clip_feature=None,
    action=None,
    state=None,
    embodiment_id=None,
):
    r"""
    Forward pass through the diffusion model

    Args:
        x (List[Tensor]):
            List of input video tensors, each with shape [C_in, F, H, W]
        t (Tensor):
            Diffusion timesteps tensor of shape [B]
        context (List[Tensor]):
            List of text embeddings each with shape [L, C]
        seq_len (`int`):
            Maximum sequence length for positional encoding
        clip_fea (Tensor, *optional*):
            CLIP image features for image-to-video mode
        y (List[Tensor], *optional*):
            Conditional video inputs for image-to-video mode, same shape as x

    Returns:
        List[Tensor]:
            List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
    """
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
        sinusoidal_embedding_1d,
    )

    if self.model_type == 'i2v':
        assert clip_feature is not None and y is not None

    # Concat [x; y] only when pretrained that way (14B). 5B uses latent only, first-frame via CLIP.
    if y is not None and self.concat_first_frame_latent:
        x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

    # embeddings
    x = self.patch_embedding(x.contiguous(memory_format=torch.channels_last_3d))

    grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
    freqs = self._create_freqs(
        grid_size=grid_size,
        start_frame=0,
    )

    x = x.flatten(start_dim=2).transpose(1, 2)
    assert x.shape[1] == seq_len

    B = x.shape[0]
    F = timestep.shape[1]

    # time embeddings
    if action is not None:
        embodiment_id = torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
        action_features = self.action_encoder(action, timestep_action, embodiment_id)
        action_length = action_features.shape[1]
        state_features = self.state_encoder(state, embodiment_id)
        action_register = torch.cat([action_features, state_features], dim=1)
        action_register_length = action_register.shape[1]
        x = torch.cat([x, action_register], dim=1)
    else:
        action_features = None
        action_length = None
        state_features = None
        action_register = None
        action_register_length = None

    # time embeddings
    timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
    timestep_original = timestep.clone()

    if action is not None:
        assert timestep_action is not None
        assert state_features is not None
        stride = timestep_action.shape[1] // state_features.shape[1]
        timestep_state = timestep_action[:, ::stride]
        timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
    e = e.unflatten(dim=0, sizes=(B, -1))
    e0 = self.time_projection(e)
    e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

    # context
    assert context.shape[1] == self.text_len
    context = self.text_embedding(context)

    if clip_feature is not None:
        clip_embedding = self.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    if clean_x is not None:
        if y is not None and self.concat_first_frame_latent:
            clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
        clean_x = self.patch_embedding(clean_x.contiguous(memory_format=torch.channels_last_3d))
        clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
        assert clean_x.shape[1] == seq_len

        x = torch.cat([clean_x, x], dim=1)

        if aug_t is None:
            aug_t = torch.zeros_like(timestep_original)
        assert aug_t is not None

        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
        e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
        e0_clean = self.time_projection(e_clean)
        e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)

    # arguments
    kwargs = dict(
        e=e0,
        freqs=freqs,
        freqs_action=self.freqs_action,
        freqs_state=self.freqs_state,
        action_register_length=action_register_length,
        context=context,
        is_tf=clean_x is not None,
    )

    # Activation checkpointing is cancelled: each block runs once, with no
    # recompute in backward. A block returns `(x, updated_kv_cache)`; the
    # kv cache is always None on the training path.
    for block in self.blocks:
        x, updated_kv_cache = block(x, **kwargs)
        assert updated_kv_cache is None

    if clean_x is not None:
        x = x[:, clean_x.shape[1]:]

    if action is not None:
        action_noise_pred = x[:, seq_len: seq_len + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
    else:
        action_noise_pred = None

    # Build a tensor that contains only video tokens per sample with length = max(video_lens)
    x_video = x[:, :seq_len]
    e_video = e[:, :seq_len]

    # Unpatchify video-only tokens
    x_video = self.head(x_video, e_video.unsqueeze(2))
    video_noise_pred = self.unpatchify(x_video, grid_size)

    return video_noise_pred, action_noise_pred
