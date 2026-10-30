# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the BEVDetOCC training loss path on accelerator inputs."""

import functools
import os


def compile_occ_train_wrapper(original, options):
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    import torch

    compile_kwargs = {"mode": options["mode"]} if "mode" in options else {}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, img_feats, *args, **kwargs):
        if not isinstance(img_feats, torch.Tensor) or img_feats.device.type != "cuda":
            return original(self, img_feats, *args, **kwargs)
        return compiled(self, img_feats, *args, **kwargs)

    return wrapped
