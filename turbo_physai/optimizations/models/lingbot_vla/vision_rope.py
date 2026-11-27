# Copyright (c) 2023, Tri Dao
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Modified by Hygon. `apply_rotary_pos_emb_flashatt` is derived from
# lingbotvla/models/vla/pi0/qwenvl_in_vla.py at lingbot-vla commit
# 4eb34b7693a0565c67433f8fac9c59a2e67eb60b (Apache-2.0).
# The fused Triton rotary path (`_apply_rotary`, `_ApplyRotaryEmb`) is derived
# from flash-attention `flash_attn/ops/triton/rotary.py` at commit
# 060c9188beec3a8b62b33a3bfa6d5d2d44975fab (v2.8.3, BSD-3-Clause, see
# third_party/licenses/flash-attention-BSD-3-Clause.txt).

from __future__ import annotations

import torch
import triton
from flash_attn.ops.triton.rotary import rotary_kernel


def _apply_rotary(x, cos, sin, conjugate=False):
    batch, seqlen, nheads, headdim = x.shape
    seqlen_ro, rotary_dim = cos.shape
    rotary_dim *= 2

    out = torch.empty_like(x)
    block_k = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    block_m = 8
    grid = lambda meta: (triton.cdiv(seqlen, meta["BLOCK_M"]), batch, nheads)
    rotary_kernel[grid](
        out,
        x,
        cos,
        sin,
        None,
        0,
        seqlen,
        rotary_dim,
        seqlen_ro,
        out.stride(0),
        out.stride(-3),
        out.stride(-2),
        out.stride(-1),
        x.stride(0),
        x.stride(-3),
        x.stride(-2),
        x.stride(-1),
        block_k,
        False,
        False,
        False,
        conjugate,
        block_m,
    )
    return out


class _ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _apply_rotary(x, cos, sin)

    @staticmethod
    def backward(ctx, grad):
        cos, sin = ctx.saved_tensors
        return _apply_rotary(grad.contiguous(), cos, sin, conjugate=True), None, None


def apply_rotary_pos_emb_flashatt(q, k, cos, sin):
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    return _ApplyRotaryEmb.apply(q, cos, sin), _ApplyRotaryEmb.apply(k, cos, sin)
