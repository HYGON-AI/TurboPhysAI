# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import gc
import logging
import time

logger = logging.getLogger(__name__)

FREEZE_AFTER = 30

_frozen = False
_refrozen = False
_sync_calls = 0
_last_collect = 0.0


def wrap_build_activation_offloading_context(original, _options=None):
    @functools.wraps(original)
    def build_activation_offloading_context(*args, **kwargs):
        global _frozen
        if not _frozen:
            gc.freeze()
            gc.set_threshold(20000, 10, 10)
            _frozen = True
            logger.info(
                "python gc frozen: %d objects, threshold %s",
                gc.get_freeze_count(),
                gc.get_threshold(),
            )
        return original(*args, **kwargs)

    return build_activation_offloading_context


def wrap_synchronize(original, _options=None):
    @functools.wraps(original)
    def synchronize(*args, **kwargs):
        global _sync_calls, _refrozen, _last_collect
        now = time.perf_counter()
        if now - _last_collect > 0.01:
            gc.collect(0)
            _last_collect = now
            _sync_calls += 1
            if _sync_calls % 50 == 0:
                gc.collect(1)
            if not _refrozen and _sync_calls >= FREEZE_AFTER:
                gc.collect()
                gc.freeze()
                _refrozen = True
                logger.info(
                    "python gc re-frozen after %d steps: %d objects",
                    _sync_calls,
                    gc.get_freeze_count(),
                )
        return original(*args, **kwargs)

    return synchronize
