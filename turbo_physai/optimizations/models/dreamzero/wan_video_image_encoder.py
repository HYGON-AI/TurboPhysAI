"""Wan image encoder (CLIP ViT): torch.compile the ViT forward pass.

"""

from __future__ import annotations

import functools

import torch

_COMPILE_DEFAULTS = {"dynamic": True}


def compile_forward(original, options):
    compile_kwargs = dict(_COMPILE_DEFAULTS)
    compile_kwargs.update(options or {})
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def forward(self, x, interpolation=False, use_31_block=False):
        return compiled(self, x, interpolation, use_31_block)

    return forward
