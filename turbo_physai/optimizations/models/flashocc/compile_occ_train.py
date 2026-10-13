"""Compile the BEVDetOCC training loss path on accelerator inputs."""

import functools


def compile_occ_train_wrapper(original, options):
    import torch

    compile_kwargs = {"mode": options["mode"]} if "mode" in options else {}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, img_feats, *args, **kwargs):
        if not isinstance(img_feats, torch.Tensor) or img_feats.device.type != "cuda":
            return original(self, img_feats, *args, **kwargs)
        return compiled(self, img_feats, *args, **kwargs)

    return wrapped
