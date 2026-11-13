# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""flex attention 的 kernel tile 调优"""

from __future__ import annotations

import functools

DEFAULT_KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 32,
    "num_warps": 4,
    "num_stages": 1,
    "bwd_BLOCK_M1": 16,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
}


def flex_attention_kernel_options(original, _options=None):
    """用固定表覆盖每次调用传入的 kernel_options。"""
    table = dict(DEFAULT_KERNEL_OPTIONS)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        caller = kwargs.get("kernel_options") or {}
        kwargs["kernel_options"] = {**caller, **table}
        return original(*args, **kwargs)

    return wrapped
