"""Independent DDP static-graph/communication fast path replacement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .ddp_state import register_static


def wrap_ddp_static_fast_path(original, options: Mapping[str, Any]):
    return register_static(original, options)
