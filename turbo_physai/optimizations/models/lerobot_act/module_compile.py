# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the original ACT VAE encoder and Transformer encoder/decoder forwards."""

import functools
import os

import torch


def _compile_forward(original):
    return torch.compile(
        original,
        dynamic=False,
        fullgraph=True,
        mode="max-autotune-no-cudagraphs",
    )


def _compile_enabled(options):
    if options:
        raise ValueError("lerobot_act.module_compile does not accept options")
    return os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") != "1"


def wrap_encoder(original, options):
    if not _compile_enabled(options):
        return original
    vae_compiled = None
    main_compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal vae_compiled, main_compiled
        if self.is_vae_encoder:
            if vae_compiled is None:
                vae_compiled = _compile_forward(original)
            return vae_compiled(self, *args, **kwargs)
        if main_compiled is None:
            main_compiled = _compile_forward(original)
        return main_compiled(self, *args, **kwargs)

    return wrapped


def wrap_decoder(original, options):
    if not _compile_enabled(options):
        return original
    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = _compile_forward(original)
        return compiled(self, *args, **kwargs)

    return wrapped
