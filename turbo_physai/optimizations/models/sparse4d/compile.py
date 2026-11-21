# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import os

from ....engine.contracts import to_primitive


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


def compile_target_wrapper(original, options):
    """Compile a target method with shared and method-specific options."""
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    compile_kwargs = to_primitive(options)
    method_options = compile_kwargs.pop("methods", {})
    compile_kwargs.update(method_options.get(original.__name__, {}))
    return torch.compile(original, **compile_kwargs)
