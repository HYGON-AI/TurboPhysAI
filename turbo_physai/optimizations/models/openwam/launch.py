# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare backend and ZeRO settings at Accelerate's existing launch boundary."""

from __future__ import annotations

import functools
import sys

from .settings import enabled, initialize


def configure_backends():
    initialize(force=True)
    from . import hw_queues, sac, vae_layout

    hw_queues.configure()
    sac.validate()
    if enabled("vae_channels_last"):
        vae_layout.validate()
    if enabled("lightop_norm"):
        from .norms import lightop_ops

        lightop_ops()
    # Source dispatcher decorators ran while all settings were inactive.
    if enabled("global_compile"):
        for name, module in tuple(sys.modules.items()):
            if name.startswith("turbo_physai.optimizations.models.openwam"):
                for function in tuple(vars(module).values()):
                    if getattr(function, "__openwam_dispatcher__", False):
                        sac.dispatcher(function)


def deepspeed_post_init(original, options):
    del options  # Tuning comes from the installed training Group wrappers.

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        configure_backends()
        result = original(self, *args, **kwargs)
        from . import configure_zero

        configure_zero(self.deepspeed_config)
        return result

    return wrapped
