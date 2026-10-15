"""Independent fused AdamW replacement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


_FUSED_ADAMW_ENABLED = False


def wrap_fused_adamw(original, options: Mapping[str, Any]):
    global _FUSED_ADAMW_ENABLED
    fused = bool(options.get("fused", True))
    _FUSED_ADAMW_ENABLED = fused

    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["fused"] = fused
        return original(*args, **kwargs)

    return wrapped


def fused_adamw_enabled() -> bool:
    return _FUSED_ADAMW_ENABLED
