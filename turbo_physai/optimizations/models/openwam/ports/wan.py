# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import types
import torch
import torch.nn as nn
import openwam.model.video_backbone.wan_backbone as _source
from turbo_physai.optimizations.common.python.source_binding import bind_missing_globals
from turbo_physai.optimizations.models.openwam.host_async import rope_freqs
from turbo_physai.optimizations.models.openwam.linear_bias2d import (
    cross_attention_kwargs,
    video_ffn,
    video_linear,
)


def WanBase_prepare(self, **kw) -> BlockLoopState:
    dit = self.dit
    motion_controller = self.motion_controller
    vace = self.vace
    latents = kw["latents"]
    timestep = kw["timestep"]
    context = kw["context"]
    context_mask = kw.get("context_mask")
    seq_lens = kw.get("seq_lens")
    clip_feature = kw.get("clip_feature")
    image_cond_latents = kw.get("y")
    vace_context = kw.get("vace_context")
    motion_bucket_id = kw.get("motion_bucket_id")
    control_camera_latents_input = kw.get("control_camera_latents_input")
    fuse_vae_embedding_in_latents = kw.get("fuse_vae_embedding_in_latents", False)
    num_clean_prefix_frames = kw.get("num_clean_prefix_frames", 0)
    use_gradient_checkpointing = kw.get("use_gradient_checkpointing", False)
    use_gradient_checkpointing_offload = kw.get(
        "use_gradient_checkpointing_offload", False
    )
    force_per_token_t_mod = bool(kw.get("force_per_token_t_mod", False))

    time_embed, time_modulation = wan_dit_forward.build_time_modulation(
        dit,
        timestep,
        latents,
        patch_size=self._dit_patch_size,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        force_per_token_t_mod=force_per_token_t_mod,
        num_clean_prefix_frames=num_clean_prefix_frames,
        zero_clean_prefix_t_mod=bool(kw.get("zero_clean_prefix_t_mod", False)),
        has_first_frame_latents=kw.get("first_frame_latents") is not None,
    )

    if motion_bucket_id is not None and motion_controller is not None:
        motion_term = motion_controller(motion_bucket_id).unflatten(
            1, (6, dit.dim)
        )  # (B, 6, dim)
        if time_modulation.dim() == 4:
            # Broadcast (B, 6, dim) across L; without the unsqueeze the add
            # right-aligns and aliases B onto L (mis-broadcasts when B == L).
            motion_term = motion_term.unsqueeze(1)  # (B, 1, 6, dim)
        time_modulation = time_modulation + motion_term
    context = dit.text_embedding(context)
    if context_mask is None:
        if seq_lens is not None:
            seq_lens = seq_lens.to(device=context.device)
            positions = torch.arange(context.shape[1], device=context.device).unsqueeze(
                0
            )
            context_mask = positions < seq_lens.unsqueeze(1)
    else:
        context_mask = context_mask.to(device=context.device, dtype=torch.bool)
        if context_mask.ndim != 2:
            raise ValueError(
                f"context_mask must be 2D [B, L], got shape {tuple(context_mask.shape)}"
            )
        if (
            context_mask.shape[0] != context.shape[0]
            or context_mask.shape[1] != context.shape[1]
        ):
            raise ValueError(
                f"context_mask shape must match context [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
            )

    hidden_states = latents
    if hidden_states.shape[0] != context.shape[0]:
        hidden_states = torch.concat([hidden_states] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    if image_cond_latents is not None and dit.require_vae_embedding:
        hidden_states = torch.cat([hidden_states, image_cond_latents], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        if context_mask is not None:
            clip_mask = torch.ones(
                (context_mask.shape[0], clip_embdding.shape[1]),
                dtype=torch.bool,
                device=context_mask.device,
            )
            context_mask = torch.cat([clip_mask, context_mask], dim=1)

    hidden_states = dit.patchify(hidden_states, control_camera_latents_input)

    grid_frames, grid_height, grid_width = hidden_states.shape[2:]
    hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c").contiguous()
    freqs = rope_freqs(
        dit,
        (grid_frames, grid_height, grid_width),
        hidden_states.device,
        lambda: (
            torch.cat(
                [
                    dit.freqs[0][:grid_frames]
                    .view(grid_frames, 1, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[1][:grid_height]
                    .view(1, grid_height, 1, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                    dit.freqs[2][:grid_width]
                    .view(1, 1, grid_width, -1)
                    .expand(grid_frames, grid_height, grid_width, -1),
                ],
                dim=-1,
            ).reshape(grid_frames * grid_height * grid_width, 1, -1)
        ),
    )

    extras = {
        "dit": dit,
        "vace": vace,
        "time_embed": time_embed,  # Wan head time embedding; consumed in finalize()
    }
    vace_hints = None
    if vace_context is not None:
        vace_hints = vace(
            hidden_states,
            vace_context,
            context,
            time_modulation,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    return BlockLoopState(
        hidden_states=hidden_states,
        time_mod=time_modulation,
        rope_freqs=freqs,
        context=context,
        context_mask=context_mask,
        grid_frames=grid_frames,
        grid_height=grid_height,
        grid_width=grid_width,
        vace_hints=vace_hints,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        extras=extras,
    )


def WanBase_pre_attn_at_layer_for_compile(
    self, layer_id: int, state: BlockLoopState, *, block=None
) -> Tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
    """Compile-friendly Wan pre-attention half using a tensor tuple post-state."""
    if block is None:
        block = state.extras["dit"].blocks[layer_id]

    time_modulation = state.time_mod
    has_seq = time_modulation.dim() == 4
    chunk_dim = 2 if has_seq else 1
    chunks = (
        block.modulation.to(dtype=time_modulation.dtype, device=time_modulation.device)
        + time_modulation
    ).chunk(6, dim=chunk_dim)
    if has_seq:
        chunks = tuple(c.squeeze(2) for c in chunks)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    residual_x = state.hidden_states
    from turbo_physai.optimizations.models.openwam.sac import run_norm

    attn_input = modulate(
        run_norm(block.norm1, state.hidden_states), shift_msa, scale_msa
    )

    self_attn = block.self_attn
    q = run_norm(self_attn.norm_q, video_linear(self_attn.q, attn_input))
    k = run_norm(self_attn.norm_k, video_linear(self_attn.k, attn_input))
    v = video_linear(self_attn.v, attn_input)
    q = rope_apply(q, state.rope_freqs, self_attn.num_heads)
    k = rope_apply(k, state.rope_freqs, self_attn.num_heads)

    post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    return q, k, v, post_state


def WanBase_post_attn_at_layer_for_compile(
    self,
    layer_id: int,
    state: BlockLoopState,
    attn_out: Tensor,
    post_state: tuple[Tensor, ...],
    *,
    block=None,
) -> BlockLoopState:
    """Compile-friendly Wan post-attention half consuming a tensor tuple."""
    if block is None:
        block = state.extras["dit"].blocks[layer_id]
    self_attn = block.self_attn
    residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

    hidden_states = block.gate(
        residual_x, gate_msa, video_linear(self_attn.o, attn_out)
    )
    context_mask = None
    if state.context_mask is not None:
        context_mask = (
            state.context_mask.unsqueeze(1)
            .expand(-1, hidden_states.shape[1], -1)
            .unsqueeze(1)
        )
    from turbo_physai.optimizations.models.openwam.sac import (
        ffn_boundary,
        run_ffn as run_complete_ffn,
        run_norm,
    )

    hidden_states = hidden_states + block.cross_attn(
        run_norm(block.norm3, hidden_states),
        state.context,
        ctx_mask=context_mask,
        **cross_attention_kwargs(block.cross_attn),
    )
    mlp_input = modulate(run_norm(block.norm2, hidden_states), shift_mlp, scale_mlp)
    if ffn_boundary():
        mlp_output = run_complete_ffn(block.ffn, mlp_input, video=True)
    else:
        mlp_output = video_ffn(block.ffn, mlp_input)
    hidden_states = block.gate(hidden_states, gate_mlp, mlp_output)
    state.hidden_states = hidden_states

    wan_dit_forward.apply_post_block_residuals(layer_id, state)
    return state


def WanBase_preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
    """Unified train preprocessing: raw data (``frames``/``text``, optional
    ``vace_videos``/``ref_images`` in kw) → the denoising-loop input dict.
    """
    device = self.device
    dtype = self.dtype

    video_latents = kw.get("video_latents")
    if video_latents is not None:
        # Stored latents (vae_latent_cache): no pixels, no VAE.
        num_frames, height, width = kw["video_size"]
    else:
        height, width, num_frames = (
            frames[0][0].size[1],
            frames[0][0].size[0],
            len(frames[0]),
        )
    height, width, num_frames = check_resize_height_width(
        height,
        width,
        num_frames,
        height_division_factor=self._height_division_factor,
        width_division_factor=self._width_division_factor,
        time_division_factor=self._time_division_factor,
        time_division_remainder=self._time_division_remainder,
    )

    batch_size = len(frames)
    context, seq_lens = wan_encode.encode_text(
        text,
        tokenizer=self._tokenizer,
        text_encoder=self.text_encoder,
        device=self.device,
    )

    stacked_inputs = None
    if video_latents is not None:
        input_latents = video_latents.to(dtype=dtype, device=device)
    else:
        all_input_videos = []
        for clip_frames in frames:
            all_input_videos.append(
                wan_encode.preprocess_video(
                    clip_frames,
                    encoder=self.video_encoder,
                    dtype=self.dtype,
                    device=self.device,
                )
            )
        stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = wan_encode.encode_video(
            stacked_inputs, vae=self.vae, encoder=self.video_encoder
        )
        input_latents = input_latents.to(dtype=dtype, device=device)

    # Variant-specific first-frame / control conditioning lives in
    # ``self._variant``, so this method carries no per-variant ``if``.
    cond = self._variant.build_train_conditioning(
        self,
        input_latents=input_latents,
        frames=frames,
        ref_images=kw.get("ref_images"),
        vace_videos=kw.get("vace_videos"),
        stacked_inputs=stacked_inputs,
        B=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        first_frame_image=kw.get("first_frame_image"),
    )

    return {
        "input_latents": input_latents,
        "context": context,
        "seq_lens": seq_lens,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "vace_context": cond.get("vace_context"),
        "fuse_vae_embedding_in_latents": cond.get(
            "fuse_vae_embedding_in_latents", False
        ),
        "num_clean_prefix_frames": cond.get("num_clean_prefix_frames", 0),
        "first_frame_latents": cond.get("first_frame_latents"),
        "clip_feature": cond.get("clip_feature"),
        "y": cond.get("y"),
    }


bind_missing_globals(globals(), _source)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

WanBase_prepare_wrapper = _port_wrapper(
    WanBase_prepare,
    "openwam.model.video_backbone.wan_backbone.WanBase.prepare",
    "openwam.compute",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

WanBase_pre_attn_at_layer_for_compile_wrapper = _port_wrapper(
    WanBase_pre_attn_at_layer_for_compile,
    "openwam.model.video_backbone.wan_backbone.WanBase.pre_attn_at_layer_for_compile",
    "openwam.compute",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

WanBase_post_attn_at_layer_for_compile_wrapper = _port_wrapper(
    WanBase_post_attn_at_layer_for_compile,
    "openwam.model.video_backbone.wan_backbone.WanBase.post_attn_at_layer_for_compile",
    "openwam.compute",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

WanBase_preprocess_input_for_train_wrapper = _port_wrapper(
    WanBase_preprocess_input_for_train,
    "openwam.model.video_backbone.wan_backbone.WanBase.preprocess_input_for_train",
    "openwam.compute",
)
