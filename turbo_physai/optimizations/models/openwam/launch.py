# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare backend and ZeRO settings at Accelerate's existing launch boundary."""

from __future__ import annotations

import functools
import os
import sys

from .settings import enabled, initialize


def configure_backends():
    initialize(force=True)
    from . import hw_queues, sac, vae_layout

    hw_queues.configure()
    sac.validate()
    if enabled("vae_channels_last"):
        vae_layout.validate()
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NDHWC", "1")
    if enabled("lightop_norm"):
        import torch

        if getattr(torch.version, "hip", None):
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


def deepspeed_mixed_precision(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, mixed_precision, *args, **kwargs):
        # The reference owner-reduction path communicates BF16 gradients in
        # BF16. Set this before Accelerate can insert its newer FP32 default.
        # An explicitly supplied communication dtype remains authoritative.
        config = self.deepspeed_config
        if (
            mixed_precision == "bf16"
            and enabled("zero_reduce_scatter")
            and config.get("zero_optimization", {}).get("stage") == 2
        ):
            config.setdefault("communication_data_type", "bf16")
        return original(self, mixed_precision, *args, **kwargs)

    return wrapped
