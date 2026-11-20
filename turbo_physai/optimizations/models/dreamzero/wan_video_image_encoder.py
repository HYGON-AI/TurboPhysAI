"""Wan image encoder (CLIP ViT): torch.compile the ViT forward pass.

"""

from __future__ import annotations

import functools
import os

_COMPILE_DEFAULTS = {"dynamic": True}


def compile_forward(original, options):
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    import torch

    compile_kwargs = {**_COMPILE_DEFAULTS, **(options or {})}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped
