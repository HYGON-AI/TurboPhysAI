# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Require NUMA affinity to be set before Torch starts each training rank."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import wraps
from typing import Any


def wrap_train_loop_numa(original, options: Mapping[str, Any]):
    """Guard DDP setup against falling back to post-Torch NUMA binding."""
    del options

    @wraps(original)
    def wrapped(*args, **kwargs):
        if os.environ.get("OPENPI_TURBO_NUMA_PREBOUND") != "1":
            raise RuntimeError(
                "openpi.numa must bind the rank in sitecustomize.py before Torch import; "
                "refusing to bind after Torch initialization"
            )
        return original(*args, **kwargs)

    return wrapped


# The active catalog targets scripts.train_pytorch.setup_ddp.
wrap_process_group_numa = wrap_train_loop_numa
