# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Video teacher：DINOv3 packed block-causal attention 走 flash-attn 2 的带 mask 变长 kernel。"""

from __future__ import annotations

import functools
from typing import Dict, List, Tuple

import torch
import torch._dynamo
from torch import Tensor

_MASK_CACHE: Dict[Tuple[int, int, str], Tensor] = {}
_EXPANDED_MASK_CACHE: Dict[Tuple[int, int, int, int, str], Tensor] = {}


def wrap_teacher_init(original, _options=None):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.attention_mode = "sdpa_block_causal"

    return wrapped


@torch.library.custom_op(
    "turbophysai_lingbot_vla::fa2_varlen_mask", mutates_args=()
)
def _fa2_varlen_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> torch.Tensor:
    """把 HCU 的带 mask 变长 kernel 包成 custom op"""
    from flash_attn import flash_attn_varlen_with_mask_func

    return flash_attn_varlen_with_mask_func(
        q,
        k,
        v,
        mask,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale=scale,
    )


@_fa2_varlen_mask.register_fake
def _fa2_varlen_mask_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> torch.Tensor:
    return torch.empty_like(q)


def _frame_causal_mask(frame_size: int, crop_len: int, device: torch.device) -> Tensor:
    key = (frame_size, crop_len, str(device))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        idx = torch.arange(crop_len, device=device)
        frame = idx // frame_size
        mask = frame.view(1, -1) <= frame.view(-1, 1)
        _MASK_CACHE[key] = mask
    return mask


@torch._dynamo.disable
def _expanded_causal_mask(
    frame_size: int,
    crop_len: int,
    batch: int,
    heads: int,
    device: torch.device,
) -> Tensor:
    key = (frame_size, crop_len, batch, heads, str(device))
    cached = _EXPANDED_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    mask = (
        _frame_causal_mask(frame_size, crop_len, device)
        .view(1, 1, crop_len, crop_len)
        .expand(batch, heads, -1, -1)
    )
    _EXPANDED_MASK_CACHE[key] = mask
    return mask


def flash_attention_block_causal(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    shapes: List[Tuple[int, int, int]],
    crops_per_shape: List[int],
    n_prefix: int,
    masks: List[Tensor],
) -> Tensor:
    if (
        not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise RuntimeError(
            "lingbot-vla-v2.video-teacher-fa2: flash-attn 2 cannot handle this attention input"
        )

    from flash_attn import flash_attn_varlen_with_mask_func  # noqa: F401 - availability check

    H, D = q.shape[1], q.shape[2]
    group_token_counts = []
    for (T, Hp, Wp), n_crops in zip(shapes, crops_per_shape):
        frame_size = n_prefix + Hp * Wp
        group_token_counts.append(n_crops * T * frame_size)

    q_groups = q.split(group_token_counts, dim=0)
    k_groups = k.split(group_token_counts, dim=0)
    v_groups = v.split(group_token_counts, dim=0)

    out_groups = []
    for group_idx, ((T, Hp, Wp), B_i) in enumerate(zip(shapes, crops_per_shape)):
        frame_size = n_prefix + Hp * Wp
        crop_len = T * frame_size
        mask = _expanded_causal_mask(frame_size, crop_len, B_i, H, q.device)
        cu_seqlens = torch.arange(
            0,
            (B_i + 1) * crop_len,
            crop_len,
            dtype=torch.int32,
            device=q.device,
        )
        out_g = _fa2_varlen_mask(
            q_groups[group_idx].contiguous(),
            k_groups[group_idx].contiguous(),
            v_groups[group_idx].contiguous(),
            mask,
            cu_seqlens,
            crop_len,
            D ** -0.5,
        )
        out_groups.append(out_g)
    return torch.cat(out_groups, dim=0)
