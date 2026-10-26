# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep MMDetection training DataLoader workers alive between epochs."""

import functools
import sys


def persistent_workers_wrapper(original, options):
    del options
    train_module = sys.modules[original.__module__]

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        original_builder = train_module.build_mmdet_dataloader

        @functools.wraps(original_builder)
        def build_with_persistent_workers(*builder_args, **builder_kwargs):
            workers = builder_kwargs.get(
                "workers_per_gpu",
                builder_args[2] if len(builder_args) > 2 else 0,
            )
            if workers > 0:
                builder_kwargs["persistent_workers"] = True
            return original_builder(*builder_args, **builder_kwargs)

        train_module.build_mmdet_dataloader = build_with_persistent_workers
        try:
            return original(*args, **kwargs)
        finally:
            train_module.build_mmdet_dataloader = original_builder

    return wrapped
