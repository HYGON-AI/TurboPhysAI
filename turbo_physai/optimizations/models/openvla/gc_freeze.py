# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import gc
from collections.abc import Mapping
from typing import Any, Callable, Optional


def _log(message: str) -> None:
    """Best-effort rank-zero log; never affects the optimization itself."""

    try:
        import overwatch

        if overwatch.is_rank_zero():
            overwatch.info(message)
    except Exception:  # noqa: BLE001 - logging must never break training
        pass


def freeze_gc() -> int:
    """Move everything currently tracked into the permanent generation.

    Returns the number of objects now parked there.  Safe to call more than
    once: ``gc.freeze()`` is additive, and a later call additionally parks
    whatever was created in the meantime.
    """

    gc.freeze()
    return gc.get_freeze_count()


def gc_freeze_wrapper(
    original: Callable, options: Optional[Mapping[str, Any]] = None
) -> Callable:

    del options

    @functools.wraps(original)
    def run_vla_training(self, *args, **kwargs):
        frozen = freeze_gc()
        _log(
            "Python GC policy ENABLED =>> freeze=1; "
            f"{frozen} objects moved to the permanent generation before the training loop"
        )
        return original(self, *args, **kwargs)

    return run_vla_training
