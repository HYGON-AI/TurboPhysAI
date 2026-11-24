# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Nonblocking H2D configuration for ACT."""

from __future__ import annotations

import functools
import inspect
from dataclasses import replace


def nonblocking_h2d_wrapper(original, options):
    """Enable non-blocking H2D for ACT's default DataLoader configuration."""
    del options
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        from accelerate.utils import DataLoaderConfiguration

        bound = signature.bind(*args, **kwargs)
        config = bound.arguments.get('dataloader_config')
        bound.arguments['dataloader_config'] = (
            DataLoaderConfiguration(non_blocking=True)
            if config is None else replace(config, non_blocking=True)
        )
        return original(*bound.args, **bound.kwargs)

    return wrapped
