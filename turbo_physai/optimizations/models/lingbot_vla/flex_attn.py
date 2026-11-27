# Copyright 2024 The HuggingFace Inc. team
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon. `flex_attention_forward` is derived from
# lingbotvla/models/vla/pi0/flex_attention.py at lingbot-vla commit
# 4eb34b7693a0565c67433f8fac9c59a2e67eb60b.

from __future__ import annotations

import functools
import logging
import os

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import _round_up_to_multiple, create_block_mask

from lingbotvla.models.vla.pi0.flex_attention import (
    _flex_attention,
    flex_attention as _eager_flex_attention,
)

logger = logging.getLogger(__name__)

_FLEX_ATTENTION = (
    _eager_flex_attention
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1"
    else _flex_attention
)

_BLOCK_SIZE = 64

_KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "num_warps": 4,
    "num_stages": 1,
    "bwd_BLOCK_M1": 16,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 64,
    "bwd_BLOCK_N2": 16,
}

_PRECISION = "tf32"
_CACHE_TAG = "flex-attn-tf32"


def _build_block_mask(mask_3d: torch.Tensor, q_len: int, kv_len: int):
    q_rounded = _round_up_to_multiple(q_len, _BLOCK_SIZE)
    kv_rounded = _round_up_to_multiple(kv_len, _BLOCK_SIZE)

    padded = mask_3d
    if q_rounded != q_len or kv_rounded != kv_len:
        padded = F.pad(
            padded,
            (0, kv_rounded - kv_len, 0, q_rounded - q_len),
            value=False,
        )

    def mask_mod(b, h, q_idx, kv_idx):
        return padded[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod,
        B=mask_3d.shape[0],
        H=1,
        Q_LEN=q_rounded,
        KV_LEN=kv_rounded,
        BLOCK_SIZE=_BLOCK_SIZE,
        device=mask_3d.device,
        _compile=False,
    )


def flex_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling=None,
):
    batch_size, seq_len, num_att_heads, head_dim = query_states.shape
    original_dtype = query_states.dtype

    query_states = query_states.transpose(1, 2).to(torch.float32)
    key_states = key_states.transpose(1, 2).to(torch.float32)
    value_states = value_states.transpose(1, 2).to(torch.float32)

    kv_len = key_states.shape[2]
    block_mask = None

    if attention_mask is not None:
        mask_3d = attention_mask[:, :, :kv_len]
        q_len = mask_3d.shape[1]

        q_pad = _round_up_to_multiple(q_len, _BLOCK_SIZE) - q_len
        kv_pad = _round_up_to_multiple(kv_len, _BLOCK_SIZE) - kv_len
        if q_pad > 0:
            query_states = F.pad(query_states, (0, 0, 0, q_pad), value=0.0)
        if kv_pad > 0:
            key_states = F.pad(key_states, (0, 0, 0, kv_pad), value=0.0)
            value_states = F.pad(value_states, (0, 0, 0, kv_pad), value=0.0)

        block_mask = _build_block_mask(mask_3d != 0, q_len, kv_len)

    attn_output = _FLEX_ATTENTION(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        kernel_options=dict(_KERNEL_OPTIONS),
        enable_gqa=True,
        scale=head_dim**-0.5 if scaling is None else scaling,
    )

    attn_output = attn_output[:, :, :seq_len, :].to(dtype=original_dtype)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(
        batch_size,
        -1,
        attn_output.shape[2] * attn_output.shape[3],
    )
    return attn_output


def _tag_compile_cache(tag: str) -> None:
    import torch.compiler.config as compiler_config

    current = str(getattr(compiler_config, "cache_key_tag", "") or "")
    if tag in current:
        return
    merged = f"{current}-{tag}" if current else tag
    compiler_config.cache_key_tag = merged
    logger.info("torch.compiler.config.cache_key_tag: %r -> %r", current, merged)


def wrap_get_float32_precision(original, _options=None):
    _tag_compile_cache(_CACHE_TAG)

    @functools.wraps(original)
    def get_float32_precision():
        return f"'{_PRECISION}'"

    logger.info("flex attention fp32 dot: ieee -> tf32 (gfx936 v_mmac tf32)")
    return get_float32_precision
