"""torch.compile wrapper for the 3D anchor grid generator."""

from functools import wraps
import os

import torch


def compile_anchor_grid_wrapper(original, options):
    """Mirror the optimized branch's @torch.compile on grid_anchors."""
    compile_options = dict((options or {}).get("compile_options", {}))
    compile_options.setdefault("backend", "inductor")
    compiled = None

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
            return original(self, *args, **kwargs)
        device = kwargs.get("device")
        if device is None:
            device = args[1] if len(args) > 1 else "cuda"
        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            return original(self, *args, **kwargs)
        if compiled is None:
            compiled = torch.compile(original, **compile_options)
        return compiled(self, *args, **kwargs)

    return wrapped
