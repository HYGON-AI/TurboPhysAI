# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""head 无关的 mask 只存一份，BlockMask 用 H=1 广播到所有 query head。"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F


_BLOCK_SIZE = 64


@torch.compiler.disable
def build_block_mask(attention_mask_3d, num_heads, q_len, kv_len, block_size=_BLOCK_SIZE):
    from torch.nn.attention.flex_attention import create_block_mask

    q_rounded = -(-q_len // block_size) * block_size
    kv_rounded = -(-kv_len // block_size) * block_size

    mask = attention_mask_3d
    if q_rounded != q_len or kv_rounded != kv_len:
        mask = F.pad(mask, (0, kv_rounded - kv_len, 0, q_rounded - q_len), value=False)

    def mask_mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=attention_mask_3d.shape[0],
        H=1,
        Q_LEN=q_rounded,
        KV_LEN=kv_rounded,
        BLOCK_SIZE=block_size,
        device=attention_mask_3d.device,
        _compile=False,
    )


def wrap_build_block_mask(original, _options=None):
    """block_size 默认 64，调用方显式传入时以传入值为准。"""
    default_size = _BLOCK_SIZE

    @functools.wraps(original)
    def wrapped(attention_mask_3d, num_heads, q_len, kv_len, block_size=None):
        return build_block_mask(
            attention_mask_3d,
            num_heads,
            q_len,
            kv_len,
            default_size if block_size is None else block_size,
        )

    return wrapped
