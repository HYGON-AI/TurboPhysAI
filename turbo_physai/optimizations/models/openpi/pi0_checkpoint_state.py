"""Process-local state shared by the two PI0 checkpoint replacements.

Turbo resolves each replacement against the original class attribute and then
installs the resulting wrapper.  The main and image groups therefore cannot
rely on wrapper nesting.  Both factories register their own options here and
return the same composite dispatcher, preserving independent group switches
without touching the official OpenPI checkout.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .common import run_checkpoint_region


_ORIGINAL: Any | None = None
_MAIN_OPTIONS: Mapping[str, Any] | None = None
_IMAGE_OPTIONS: Mapping[str, Any] | None = None


def register_original(original: Any) -> None:
    global _ORIGINAL
    if _ORIGINAL is None:
        _ORIGINAL = original


def make_dispatcher(fallback: Any):
    """Return the dispatcher installed on ``PI0Pytorch._apply_checkpoint``."""

    register_original(fallback)

    def wrapped(self, func, *args, **kwargs):
        function_name = getattr(func, "__name__", "")
        if function_name == "forward_func":
            options = _MAIN_OPTIONS
            cache_key = "pi0_main_forward"
        elif function_name == "image_embed_func":
            options = _IMAGE_OPTIONS
            cache_key = "pi0_image_embed"
        else:
            options = None
            cache_key = ""

        if (
            options is not None
            and getattr(self, "gradient_checkpointing_enabled", False)
            and self.training
        ):
            return run_checkpoint_region(
                torch.utils.checkpoint.checkpoint,
                func,
                args,
                kwargs,
                cache_key=cache_key,
                options=options,
            )

        original = _ORIGINAL if _ORIGINAL is not None else fallback
        return original(self, func, *args, **kwargs)

    return wrapped


def register_main(original: Any, options: Mapping[str, Any]):
    global _MAIN_OPTIONS
    register_original(original)
    _MAIN_OPTIONS = options
    return make_dispatcher(original)


def register_image(original: Any, options: Mapping[str, Any]):
    global _IMAGE_OPTIONS
    register_original(original)
    _IMAGE_OPTIONS = options
    return make_dispatcher(original)
