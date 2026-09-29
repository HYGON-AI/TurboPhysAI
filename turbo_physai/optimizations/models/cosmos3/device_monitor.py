# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from functools import wraps
import os
import pynvml
import torch
from cosmos_framework.utils import distributed, log


def on_train_start(self, model, iteration=0):
    torch.cuda.reset_peak_memory_stats()
    self.world_size = distributed.get_world_size()
    self.rank = distributed.get_rank()
    self._enabled = True
    config_job = self.config.job
    self.local_dir = f"{config_job.path_local}/{self.name}"
    if self.rank == 0:
        os.makedirs(self.local_dir, exist_ok=True)
        log.info(f"{self.name} callback: local_dir: {self.local_dir}")

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    try:
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
    except Exception as exc:
        self._enabled = False
        if self.rank == 0:
            log.warning(f"{self.name} callback disabled: NVML device handle unavailable: {exc}")


def guard_every_n(original, options):
    """Keep the upstream metrics implementation, skipping unavailable NVML handles."""
    @wraps(original)
    def guarded(self, *args, **kwargs):
        if not getattr(self, "_enabled", True):
            return None
        return original(self, *args, **kwargs)
    return guarded
