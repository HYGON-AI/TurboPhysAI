"""Disable activation checkpointing for FlashOCC's image backbone."""

import copy
import functools


def disable_activation_checkpoint_wrapper(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if args:
            backbone_cfg = args[0]
        else:
            backbone_cfg = kwargs.get("img_backbone")

        if not isinstance(backbone_cfg, dict):
            raise TypeError(
                "BEVDet.img_backbone must be a mapping for the checkpoint replacement"
            )

        backbone_cfg = copy.copy(backbone_cfg)
        backbone_cfg["with_cp"] = False
        if args:
            args = (backbone_cfg, *args[1:])
        else:
            kwargs = dict(kwargs)
            kwargs["img_backbone"] = backbone_cfg
        return original(self, *args, **kwargs)

    return wrapped
