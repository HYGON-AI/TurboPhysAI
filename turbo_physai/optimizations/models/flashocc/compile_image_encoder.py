"""Compile the BEVDet image encoder on accelerator inputs."""

import functools


def compile_image_encoder_wrapper(original, options):
    import torch

    compile_kwargs = {"mode": options["mode"]} if "mode" in options else {}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, img, *args, **kwargs):
        if img.device.type != "cuda":
            return original(self, img, *args, **kwargs)
        return compiled(self, img, *args, **kwargs)

    return wrapped
