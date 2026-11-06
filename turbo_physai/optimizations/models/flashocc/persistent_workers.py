# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep MMDetection training DataLoader workers alive between epochs."""

import functools
import inspect


def persistent_workers_wrapper(original, options):
    del options
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        cfg = bound.arguments["cfg"]
        cfg.data.persistent_workers = cfg.data.workers_per_gpu > 0
        return original(*args, **kwargs)

    return wrapped
