# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Enable pinned host memory for MMDetection DataLoader instances."""

import functools


def pin_memory_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        kwargs["pin_memory"] = True
        return original(*args, **kwargs)

    return wrapped
