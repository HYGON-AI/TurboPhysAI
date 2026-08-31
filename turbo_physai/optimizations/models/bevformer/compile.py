# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""torch.compile wrappers for BEVFormer hot paths."""

import functools
import os


def _compile_mode():
    return os.getenv(
        "TURBO_PHYSAI_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
    )


def compile_wrapper(original, options):
    del options
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    compiled = torch.compile(
        original, mode=_compile_mode()
    )

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped
