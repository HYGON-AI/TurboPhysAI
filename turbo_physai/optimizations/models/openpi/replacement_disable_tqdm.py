"""Disable tqdm rendering without modifying the upstream training loop."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any


def wrap_tqdm_disabled(original, options: Mapping[str, Any]):
    # The deferred-loop replacement also consults this marker so it can avoid
    # constructing a disabled tqdm object at all, matching the reference loop.
    import os

    os.environ["OPENPI_TURBO_DISABLE_TQDM"] = "1"
    del options

    @wraps(original)
    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["disable"] = True
        return original(*args, **kwargs)

    return wrapped
