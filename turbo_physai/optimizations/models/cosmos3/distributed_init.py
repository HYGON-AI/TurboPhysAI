# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""utils.distributed.init with optional NVML affinity (HCU containers)."""

from __future__ import annotations

import ctypes
import os
from datetime import timedelta

import pynvml
import torch
import torch.distributed as dist

import cosmos_framework.utils.distributed as _dist_mod

from ._source_binding import bind_missing_globals

def init() -> int | None:
    """Initialize distributed training."""
    if dist.is_initialized():
        return torch.cuda.current_device()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    # NVML affinity is an optional optimization.  HCU/HIP containers often do
    # not ship libnvidia-ml.so, so a missing NVML library must not block
    # distributed initialization.
    try:
        pynvml.nvmlInit()
        device = Device(local_rank)
        os.sched_setaffinity(0, device.get_cpu_affinity())
    except Exception as exc:
        log.warning(f"Failed to set optional device affinity: {exc}")
    # Set up distributed communication. CPU checkpoint conversion needs Gloo
    # because NCCL cannot synchronize CPU-resident tokenizer or model tensors.
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "0"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if dist.is_available():
        torch.cuda.set_device(local_rank)
        # Get the timeout value from environment variable
        timeout_seconds = os.getenv("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", 1800)
        # Convert the timeout to an integer (if it isn't already) and then to a timedelta
        timeout_timedelta = timedelta(seconds=int(timeout_seconds))
        backend = "nccl" if os.environ.get("COSMOS_DEVICE", "cuda").lower() == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://", timeout=timeout_timedelta)
        log.critical(
            f"Initialized distributed training with local rank {local_rank} using {backend} with timeout {timeout_seconds}",
            rank0_only=False,
        )
    # Increase the L2 fetch granularity for faster speed.
    # For oss, we need to search for the library in site-packages.
    if INTERNAL:
        _libcudart = ctypes.CDLL("libcudart.so")
        # Set device limit on the current device.
        p_value = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(p_value, ctypes.c_int(0x05))
    log.info(f"Training with {get_world_size()} GPUs.")


bind_missing_globals(globals(), _dist_mod)
