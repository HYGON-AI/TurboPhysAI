# Copyright 2025 DiT4DiT team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Adapted from DiT4DiT d8854a1; see THIRD_PARTY_NOTICES.md.

"""Keep upstream dataset/statistics handling and tune its DataLoader factory."""

import functools
import types


def build_dataloader_wrapper(original, options):
    # A private globals mapping scopes the factory to this function: no global
    # torch/Accelerate patches, and spawned workers keep the original dataset.
    factory = original.__globals__["DataLoader"]
    settings = dict(options)

    def loader(*args, **kwargs):
        kwargs.update(settings)
        if int(kwargs.get("num_workers", 4)) == 0:
            for key in ("multiprocessing_context", "persistent_workers", "prefetch_factor", "in_order"):
                kwargs.pop(key, None)
        return factory(*args, **kwargs)

    namespace = dict(original.__globals__, DataLoader=loader)
    tuned = types.FunctionType(
        original.__code__, namespace, original.__name__,
        original.__defaults__, original.__closure__,
    )
    tuned.__kwdefaults__ = original.__kwdefaults__
    return functools.wraps(original)(tuned)


def prepare_data_wrapper(original, options):
    @functools.wraps(original)
    def wrapped(cfg, accelerator, output_dir):
        accelerator.dataloader_config.non_blocking = True
        return original(cfg, accelerator, output_dir)
    return wrapped
