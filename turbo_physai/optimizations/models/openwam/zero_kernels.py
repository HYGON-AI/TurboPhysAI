# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for opt-in ZeRO routes. Imported only on supported devices."""

import triton
import triton.language as tl


@triton.jit
def transfer_kernel(
    Src,
    Dst,
    Meta,
    Scale: tl.constexpr,
    Round: tl.constexpr,
    Reverse: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    # One descriptor per tile: source offset, destination offset, valid length.
    src = tl.load(Meta + tile * 3)
    dst = tl.load(Meta + tile * 3 + 1)
    count = tl.load(Meta + tile * 3 + 2)
    if Reverse:
        src, dst = dst, src
    x = tl.arange(0, BLOCK)
    value = tl.load(Src + src + x, x < count, other=0)
    if Round:
        # Preserve the original bucket-dtype rounding BEFORE communication.
        value = (value.to(tl.float32) * Scale).to(Src.dtype.element_ty)
    tl.store(Dst + dst + x, value, x < count)


@triton.jit
def norm_partials_kernel(
    Pointers, Meta, Partials, DTYPE: tl.constexpr, BLOCK: tl.constexpr
):
    tile = tl.program_id(0)
    index = tl.load(Meta + tile * 3)
    offset = tl.load(Meta + tile * 3 + 1)
    count = tl.load(Meta + tile * 3 + 2)
    ptr = tl.load(Pointers + index).to(tl.pointer_type(DTYPE))
    x = tl.arange(0, BLOCK)
    value = tl.load(ptr + offset + x, x < count, other=0).to(tl.float64)
    tl.store(Partials + tile, tl.sum(value * value, 0))
