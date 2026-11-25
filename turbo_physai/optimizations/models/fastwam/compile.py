# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the same four hot paths as optimized FastWAM, on first use."""

import functools
import os
import threading


def compile_hotpath(original):
    compiled = None
    lock = threading.Lock()

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal compiled
        if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
            return original(*args, **kwargs)
        if compiled is None:
            with lock:
                if compiled is None:
                    import torch

                    compiled = torch.compile(
                        original,
                        mode=os.getenv(
                            "TURBO_PHYSAI_FASTWAM_COMPILE_MODE",
                            "max-autotune-no-cudagraphs",
                        ),
                    )
        return compiled(*args, **kwargs)

    return wrapped
