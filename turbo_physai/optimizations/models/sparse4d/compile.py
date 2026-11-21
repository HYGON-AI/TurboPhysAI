# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

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


def compile_class_wrapper(original, options):
    """Apply ``torch.compile`` to the module class, matching source decorators.

    ``torch.compile(ModuleClass)`` installs the optimized call implementation
    on the class itself. Compiling ``ModuleClass.forward`` is not equivalent:
    it traces at a different boundary and creates per-instance type guards.
    """
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original
    mode = options.get("mode", "max-autotune-no-cudagraphs")
    return torch.compile(original, mode=mode)