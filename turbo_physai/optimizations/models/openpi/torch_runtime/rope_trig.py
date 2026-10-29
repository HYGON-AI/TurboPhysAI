# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Precompute the exact Triton FP32 functions used by the generated kernel."""
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import math as tl_math


@triton.jit
def precompute(freq, cos_ptr, sin_ptr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(freq + i, i < N, other=0.0)
    c = tl_math.cos(x)
    s = tl_math.sin(x)
    tl.store(cos_ptr + i, c, i < N)
    tl.store(sin_ptr + i, s, i < N)
