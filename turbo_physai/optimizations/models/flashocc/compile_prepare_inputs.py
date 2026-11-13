# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile BEVDet camera-geometry input preparation on accelerator inputs."""

import functools
import os


def compile_prepare_inputs_wrapper(original, options):
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    import torch

    compile_kwargs = {"mode": options["mode"]} if "mode" in options else {}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, inputs, *args, **kwargs):
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return original(self, inputs, *args, **kwargs)
        if inputs[0].device.type != "cuda":
            return original(self, inputs, *args, **kwargs)
        return compiled(self, inputs, *args, **kwargs)

    return wrapped
