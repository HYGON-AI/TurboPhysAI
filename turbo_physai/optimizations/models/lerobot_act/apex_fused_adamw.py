# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Use Apex FusedAdam in AdamW mode for the LeRobot ACT optimizer."""

from dataclasses import asdict
from functools import wraps


def apex_fused_adamw_wrapper(original, options):
    del options

    @wraps(original)
    def wrapped(config, params):
        from apex.optimizers import FusedAdam

        kwargs = asdict(config)
        kwargs.pop('grad_clip_norm')
        optimizer = FusedAdam(params, adam_w_mode=True, bias_correction=True, **kwargs)
        for group in optimizer.param_groups:
            group.setdefault('step', 0)
        return optimizer

    return wrapped
