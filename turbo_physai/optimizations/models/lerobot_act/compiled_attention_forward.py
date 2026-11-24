# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Triton pieces for the validated ACT 902x902 score-forward implementation."""

import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, max2


@triton.jit
def softmax_stats(x, shift, denom, prob, ROWS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0) * TILE + tl.arange(0, TILE)[:, None]
    col = tl.arange(0, 1024)[None, :]
    idx = row * 902 + col
    valid = (row < ROWS) & (col < 902)
    scores = tl.load(x + idx, valid, other=0)
    mx = max2(tl.where(valid, scores, float("-inf")), 1)[:, None]
    ex = libdevice.exp(scores - mx)
    sm = tl.sum(tl.where(valid, ex, 0.0), 1)[:, None]
    head = row // 902
    tl.store(prob + idx + head * 28, ex / sm, valid)
    tl.store(shift + row, mx, row < ROWS)
    tl.store(denom + row, sm, row < ROWS)


@triton.jit
def dropout_linear(seed, keep, prob, seed_offset, ELEMS: tl.constexpr, TILE: tl.constexpr):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    head = i // (902 * 902)
    seed_value = tl.load(seed + seed_offset)
    bits = tl.randint(seed_value, i.to(tl.uint32)).to(tl.int32, bitcast=True)
    # First integer whose original fp32 tl.rand conversion is strictly > 0.1.
    mask = (bits ^ (bits >> 31)) >= 214748393
    offset = i + head * 28
    p = tl.load(prob + offset, i < ELEMS, other=0)
    p = (mask.to(tl.float32) * p) * 1.1111111111111112
    tl.store(prob + offset, p, i < ELEMS)
    tl.store(keep + i + head * 92, mask, i < ELEMS)
