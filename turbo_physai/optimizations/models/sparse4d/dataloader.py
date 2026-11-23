# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Configure DataLoaders created through the Sparse4D dataset builder."""

from functools import wraps
from inspect import signature


def dataloader_wrapper(original, options):
    loader_signature = signature(original)

    @wraps(original, updated=())
    def wrapped(*args, **kwargs):
        bound = loader_signature.bind(*args, **kwargs)
        has_workers = bound.arguments.get("num_workers", 0) > 0
        bound.arguments["pin_memory"] = True
        bound.arguments.setdefault("persistent_workers", has_workers)
        bound.arguments.setdefault("prefetch_factor", 2 if has_workers else None)
        return original(*bound.args, **bound.kwargs)

    return wrapped
