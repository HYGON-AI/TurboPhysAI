# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""torch.compile wrappers for BEVFormer hot paths."""

import functools
import os


def compile_wrapper(original, options):
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    mode = options.get("mode", "max-autotune-no-cudagraphs")
    compiled = torch.compile(original, mode=mode)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped
