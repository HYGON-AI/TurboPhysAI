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
        dataset = bound.arguments.get("dataset")
        workers = bound.arguments.get("num_workers", 0)
        if (type(dataset).__module__.startswith("projects.mmdet3d_plugin.datasets.")
                and workers > 0):
            bound.arguments["persistent_workers"] = True
        return original(*bound.args, **bound.kwargs)

    return wrapped
