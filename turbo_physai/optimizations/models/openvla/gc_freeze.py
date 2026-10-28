# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import gc
from collections.abc import Mapping
from typing import Any, Callable, Optional


def freeze_gc() -> int:
    gc.freeze()
    return gc.get_freeze_count()


def gc_freeze_wrapper(
    original: Callable, options: Optional[Mapping[str, Any]] = None
) -> Callable:

    del options

    @functools.wraps(original)
    def run_vla_training(self, *args, **kwargs):
        freeze_gc()
        return original(self, *args, **kwargs)

    return run_vla_training
