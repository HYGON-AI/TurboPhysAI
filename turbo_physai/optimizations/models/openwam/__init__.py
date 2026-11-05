# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

"""OpenWAM model preparation and Turbo-configured optimization helpers."""

import logging
import os
from .settings import enabled, value as value
from .launch import configure_backends as configure_backends

logger = logging.getLogger(__name__)


def configure_zero(config: dict) -> None:
    if (enabled("zero_scaled_pack") or enabled("zero_norm_overlap")) and not enabled(
        "zero_reduce_scatter"
    ):
        raise ValueError(
            "ZeRO scaled packing and norm overlap require zero_reduce_scatter"
        )
    if enabled("host_async"):
        from turbo_physai.optimizations.models.openwam.host_async import (
            configure_deepspeed,
        )

        configure_deepspeed(config)
    if enabled("zero_reduce_scatter") or enabled("online_encode"):
        zero = config["zero_optimization"]
        if zero.get("stage") != 2:
            raise ValueError(
                "ReduceScatter and online encoding currently require ZeRO-2"
            )
        if enabled("zero_reduce_scatter"):
            zero.update(
                reduce_scatter=True,
                contiguous_gradients=True,
                use_multi_rank_bucket_allreduce=True,
                overlap_comm=True,
            )
    if enabled("deferred_gather") and config["zero_optimization"].get("stage") != 2:
        raise ValueError("deferred_gather currently requires ZeRO-2")
    if enabled("zero_overlap"):
        zero = config["zero_optimization"]
        zero.update(overlap_comm=True, contiguous_gradients=True)
        # Keep activation headroom while preserving explicit caller configurations.
        # Bucket size is in elements, not bytes.
        if zero.get("stage") == 2:
            zero.setdefault("reduce_bucket_size", 100_000_000)
            import torch

            # PyTorch also prioritizes CUBLASLT_WORKSPACE_SIZE on HIP. Preserve
            # either explicit override; this default is in KiB (32 MiB).
            if (
                torch.version.hip
                and os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "1"
                and "CUBLASLT_WORKSPACE_SIZE" not in os.environ
                and "HIPBLASLT_WORKSPACE_SIZE" not in os.environ
            ):
                os.environ["HIPBLASLT_WORKSPACE_SIZE"] = "32768"


def prepare_model(architecture) -> None:
    """Prepare SAC and model optimizations before DeepSpeed."""
    from turbo_physai.optimizations.models.openwam.sac import prepare

    prepare(architecture)


def prepare_runtime_constants(architecture) -> None:
    """Stage small, immutable training constants after Accelerate device placement."""
    if not enabled("constant_cache"):
        return
    backbone = getattr(architecture, "video_backbone", None)
    if backbone is None:
        return
    device = architecture.device
    for stream in (backbone, getattr(architecture, "action_backbone", None)):
        scheduler = getattr(stream, "scheduler", None)
        for name in ("timesteps", "sigmas", "linear_timesteps_weights"):
            value = getattr(scheduler, name, None)
            if value is not None:
                setattr(scheduler, name, value.to(device=device))


def stage_training_indices(indices, device):
    """Preserve CPU sampling but avoid a synchronizing transfer behind VAE work."""
    import torch

    target = torch.device(device)
    if indices.device.type == "cpu" and target.type == "cuda":
        return indices.pin_memory().to(device=target, non_blocking=True)
    return indices.to(device=target)
