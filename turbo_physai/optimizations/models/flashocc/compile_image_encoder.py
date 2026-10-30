# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the BEVDet image encoder on accelerator inputs."""

import functools
import os


def compile_image_encoder_wrapper(original, options):
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    import torch

    compile_kwargs = {"mode": options["mode"]} if "mode" in options else {}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, img, *args, **kwargs):
        if img.device.type != "cuda":
            return original(self, img, *args, **kwargs)
        return compiled(self, img, *args, **kwargs)

    return wrapped
