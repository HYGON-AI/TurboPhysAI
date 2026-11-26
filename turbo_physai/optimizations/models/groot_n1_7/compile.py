"""Optional method-level compilation wrappers."""

import os
from functools import wraps


def compile_method_wrapper(original, options):
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    import torch
    compiled = torch.compile(original, mode=options.get("mode", "max-autotune-no-cudagraphs"))

    @wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)
    return wrapped
