"""Compile the BEVDet BEV encoder on accelerator inputs."""

import functools


def compile_bev_encoder_wrapper(original, options):
    import torch

    compile_kwargs = {"mode": options["mode"]} if "mode" in options else {}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor) or x.device.type != "cuda":
            return original(self, x, *args, **kwargs)
        return compiled(self, x, *args, **kwargs)

    return wrapped
