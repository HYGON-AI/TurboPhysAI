# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Compile the ACT visual backbone."""

import functools
import os

import torch


def backbone_compile_wrapper(original, options):
    del options
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(original, dynamic=False, options={'triton.cudagraphs': False})
        return compiled(self, *args, **kwargs)

    return wrapped
