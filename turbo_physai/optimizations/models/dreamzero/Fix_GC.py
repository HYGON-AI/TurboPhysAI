# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Freeze long-lived Python objects after DreamZero's warmup microsteps."""

import functools
import gc

import torch.distributed as dist


def freeze_gc_after_warmup(original, options):
    """Wrap VLATrainer.training_step to freeze GC once per trainer."""

    @functools.wraps(original)
    def training_step(self, *args, **kwargs):
        loss = original(self, *args, **kwargs)
        # BaseTrainer increments current_step after the backward pass.
        if self.current_step >= 50 and not getattr(self, "_gc_frozen", False):
            if dist.is_initialized():
                dist.barrier()
            gc.collect(2)
            gc.freeze()
            self._gc_frozen = True
        return loss

    return training_step
