# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Qwen3-VL encoder HIP optimizations (adapted source port).

- Vision patch embed: channels-last input before the Conv3d projection
  (dense and MoE variants).
- Vision attention/blocks: precomputed ``chunk_lengths`` threading (skips a
  device sync per block on the non-FA path).
- ``Qwen3VLVisionModel``: per-``grid_thw`` aux cache (rotary emb, cu_seqlens,
  chunk lengths) with FIFO eviction; installed via ``qwen_vision_model_post_init``.
- ``get_rope_index``: full-CPU multimodal branch with one final H2D move.

The adapted source additionally ships a full local ``qwen3_vl`` package
registered with the transformers Auto classes.  Cosmos upstream already
contains ``cosmos_framework.model.generator.reasoner.qwen3_vl`` with the same
class names, so these method-level ports target that package; the extra
``AutoConfig.register(..., exist_ok=True)`` step lives in
``qwen_registration`` below and is a separate catalog member.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl as _qwen_mod
import cosmos_framework.model.generator.reasoner.qwen3_vl.utils as _qwen_utils_mod
import cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe as _qwen_moe_mod

from ._source_binding import bind_missing_globals

try:
    from cosmos_framework.utils.memory_format import to_channels_last
except ModuleNotFoundError as exc:
    if exc.name != "cosmos_framework.utils.memory_format":
        raise
    from ...common.pytorch.memory_format import to_channels_last


def qwen_vision_model_post_init(original, options):
    """Wrap ``Qwen3VLVisionModel.__init__`` to create the aux cache."""
    del options

    def initialized(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._aux_cache = {}
        self._aux_cache_capacity = 64
        return result

    return initialized


def qwen_local_registration(original, options):
    """Wrap ``HFModel.__init__``: register the framework-local Qwen3-VL classes
    with the transformers Auto classes (exist_ok) before any config resolution,
    mirroring the adapted source's import-time registration in
    ``reasoner/qwen3_vl/__init__``.
    """
    del options

    def initialized(self, *args, **kwargs):
        from transformers import AutoConfig, AutoModelForImageTextToText

        from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig,
        )
        from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        AutoConfig.register("qwen3_vl", Qwen3VLConfig, exist_ok=True)
        AutoModelForImageTextToText.register(
            Qwen3VLConfig, Qwen3VLForConditionalGeneration, exist_ok=True
        )
        return original(self, *args, **kwargs)

    return initialized


def vision_patch_embed_forward(
    self, hidden_states: torch.Tensor
) -> torch.Tensor:  # hidden_states: [N_patches,in_channels*temporal_patch_size*patch_size*patch_size]
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )  # [N_patches,in_channels,temporal_patch_size,patch_size,patch_size]
    hidden_states = to_channels_last(hidden_states)
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
        -1, self.embed_dim
    )  # [N_patches,embed_dim]
    return hidden_states  # [N_patches,embed_dim]


def moe_vision_patch_embed_forward(
    self, hidden_states: torch.Tensor
) -> torch.Tensor:  # hidden_states: [N_patches,in_channels*temporal_patch_size*patch_size*patch_size]
    target_dtype = self.proj.weight.dtype
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )  # [N_patches,in_channels,temporal_patch_size,patch_size,patch_size]
    hidden_states = to_channels_last(hidden_states)
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
        -1, self.embed_dim
    )  # [N_patches,embed_dim]
    return hidden_states


def vision_attention_forward(
    self,
    hidden_states: torch.Tensor,  # [N_vision,hidden_size]
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    chunk_lengths: Optional[list[int]] = None,
    **kwargs,
) -> torch.Tensor:  # [N_vision,hidden_size]
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )  # each: [N_vision,num_heads,head_dim]
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    # each: [N_vision,num_heads,head_dim]

    query_states = query_states.transpose(0, 1).unsqueeze(0)  # [1,num_heads,N_vision,head_dim]
    key_states = key_states.transpose(0, 1).unsqueeze(0)  # [1,num_heads,N_vision,head_dim]
    value_states = value_states.transpose(0, 1).unsqueeze(0)  # [1,num_heads,N_vision,head_dim]

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    if self.config._attn_implementation == "flash_attention_2":
        # Flash Attention 2: Use cu_seqlens for variable length attention
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        if chunk_lengths is None:
            chunk_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [
            torch.split(tensor, chunk_lengths, dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)  # [1,N_vision,num_heads,head_dim]

    attn_output = attn_output.reshape(seq_length, -1).contiguous()  # [N_vision,hidden_size]
    attn_output = self.proj(attn_output)  # [N_vision,hidden_size]
    return attn_output  # [N_vision,hidden_size]


def vision_block_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    chunk_lengths: Optional[list[int]] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        chunk_lengths=chunk_lengths,
        **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states


def _get_cached_aux(self, grid_thw: torch.Tensor):
    """Return ``(rotary_pos_emb, cu_seqlens, chunk_lengths)`` for grid_thw.

    Cached across steps by (device, grid_thw values). Skips ``.item()``
    host-device syncs in ``rot_pos_emb`` and the ``repeat_interleave +
    cumsum + pad + tolist`` chain in ``forward``. Autograd-safe: none of
    these three quantities carries learnable parameters.
    """
    if torch.jit.is_tracing():
        # Never cache under tracing/export: the traced graph must recompute.
        return _compute_aux(self, grid_thw)

    if not hasattr(self, "_aux_cache"):
        self._aux_cache = {}
        self._aux_cache_capacity = 64
    key = (grid_thw.device, tuple(grid_thw.flatten().tolist()))
    entry = self._aux_cache.get(key)
    if entry is None:
        entry = _compute_aux(self, grid_thw)
        if len(self._aux_cache) >= self._aux_cache_capacity:
            # FIFO eviction — dict preserves insertion order (Python 3.7+).
            self._aux_cache.pop(next(iter(self._aux_cache)))
        self._aux_cache[key] = entry
    return entry


def _compute_aux(self, grid_thw: torch.Tensor):
    rotary_pos_emb = self.rot_pos_emb(grid_thw).detach()  # [N_vision,head_dim//2]
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).detach()  # [N_media+1]

    chunk_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    return rotary_pos_emb, cu_seqlens, chunk_lengths


def vision_model_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)  # [N_vision,embed_dim]

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)  # [N_vision,hidden_size]
    hidden_states = hidden_states + pos_embeds  # [N_vision,hidden_size]

    rotary_pos_emb, cu_seqlens, chunk_lengths = _get_cached_aux(self, grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)  # [N_vision,hidden_size]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)  # [N_vision,head_dim//2]
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)  # [N_vision,head_dim]
    position_embeddings = (emb.cos(), emb.sin())  # each: [N_vision,head_dim]

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            chunk_lengths=chunk_lengths,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                hidden_states
            )
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)  # [N_merged,out_hidden_size]

    return hidden_states, deepstack_feature_lists  # [N_merged,out_hidden_size], list of [N_merged,out_hidden_size]


def get_rope_index(
    model: Any,
    input_ids: Optional[torch.Tensor] = None,  # [B,N]
    image_grid_thw: Optional[torch.Tensor] = None,  # [num_images,3]
    video_grid_thw: Optional[torch.Tensor] = None,  # [num_videos,3]
    attention_mask: Optional[torch.Tensor] = None,  # [B,N]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL multimodal RoPE positions and deltas.

    The multimodal branch is data-dependent (per-media scalar reads, Python
    `list.index` searches, dynamic per-media shapes), so we run it entirely on
    CPU to avoid host↔device sync barriers, then move the result to the input
    device once at the end. See `qwen3_vl.py` `get_rope_index` for the caller;
    it's called once per generation in prefill only, so the extra H2D of the
    small index tensor is negligible.
    """
    spatial_merge_size = model.config.vision_config.spatial_merge_size
    image_token_id = model.config.image_token_id
    video_token_id = model.config.video_token_id
    vision_start_token_id = model.config.vision_start_token_id

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        target_device = input_ids.device
        input_ids_cpu = input_ids.cpu()
        attention_mask_cpu = (
            torch.ones_like(input_ids_cpu) if attention_mask is None else attention_mask.cpu()
        )  # [B,N]
        image_grid_thw_cpu = image_grid_thw.cpu() if image_grid_thw is not None else None
        if video_grid_thw is not None:
            video_grid_thw_cpu = video_grid_thw.cpu()
            video_grid_thw_cpu = torch.repeat_interleave(
                video_grid_thw_cpu, video_grid_thw_cpu[:, 0], dim=0
            ).clone()  # [sum_T,3]; clone() so we don't mutate the caller's tensor via the write below
            video_grid_thw_cpu[:, 0] = 1
        else:
            video_grid_thw_cpu = None

        position_ids = torch.ones(
            3, input_ids_cpu.shape[0], input_ids_cpu.shape[1], dtype=input_ids_cpu.dtype
        )  # [3,B,N]
        mrope_position_deltas: list[int] = []
        image_index, video_index = 0, 0

        for i, sample_input_ids in enumerate(input_ids_cpu):
            sample_input_ids = sample_input_ids[attention_mask_cpu[i] == 1]  # [N_unmasked]
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]  # [N_media]
            image_nums = int((vision_tokens == image_token_id).sum())
            video_nums = int((vision_tokens == video_token_id).sum())
            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list: list[torch.Tensor] = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = image_grid_thw_cpu[image_index].tolist()
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw_cpu[video_index].tolist()
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                text_len = ed - st

                st_idx = int(llm_pos_ids_list[-1].max()) + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)  # [3,text_len]

                # Vectorized [T,H,W] index generation in one meshgrid call — replaces
                # three separate arange+view+expand+flatten chains.
                t_idx, h_idx, w_idx = torch.meshgrid(
                    torch.arange(llm_grid_t),
                    torch.arange(llm_grid_h),
                    torch.arange(llm_grid_w),
                    indexing="ij",
                )
                llm_pos_ids_list.append(
                    torch.stack([t_idx.flatten(), h_idx.flatten(), w_idx.flatten()]) + text_len + st_idx
                )  # [3,T*H*W]
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = int(llm_pos_ids_list[-1].max()) + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)  # [3,text_len]

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)  # [3,N_unmasked]
            position_ids[..., i, attention_mask_cpu[i] == 1] = llm_positions
            mrope_position_deltas.append(int(llm_positions.max()) + 1 - input_ids_cpu.shape[1])

        mrope_position_deltas_t = torch.tensor(mrope_position_deltas, dtype=torch.long).unsqueeze(1)  # [B,1]
        return position_ids.to(target_device), mrope_position_deltas_t.to(target_device)

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1  # [B,N]
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)  # [3,B,N]
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]  # [B,1]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]  # [B,1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device).view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        )  # [3,B,N]
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )  # [B,1]

    return position_ids, mrope_position_deltas


bind_missing_globals(globals(), _qwen_mod, _qwen_utils_mod, _qwen_moe_mod)
