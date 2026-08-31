# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""torch.compile wrapper for BEVFusion hot paths."""

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
    it traces at a different boundary and creates per-instance type guards in
    nested sparse modules.
    """
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original
    mode = options.get("mode", "max-autotune-no-cudagraphs")
    return torch.compile(original, mode=mode)


def compile_transfusion_forward_single_wrapper(original, options):
    """Compile the migrated TransFusion forward instead of the captured original."""

    from .transfusion import transfusion_forward_single

    # Patch planning prepares every group before applying it.  Using the
    # migrated callable explicitly keeps the later compile group from restoring
    # the original method captured during planning.
    return compile_wrapper(transfusion_forward_single, options)


def dynamo_disable_wrapper(original, options):
    """Keep a reference-marked method outside surrounding Dynamo graphs."""

    del options
    import torch

    return torch._dynamo.disable(original)
