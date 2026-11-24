# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""DDP options for ACT."""

from __future__ import annotations

import functools


def ddp_options_wrapper(original, options):
    """Match reference DDP and apply explicitly selected ACT DDP options."""
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        kwargs['find_unused_parameters'] = False
        if 'static_graph' in options:
            kwargs['static_graph'] = options['static_graph']
        return original(*args, **kwargs)

    return wrapped
