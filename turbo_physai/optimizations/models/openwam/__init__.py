# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

"""OpenWAM model preparation and Turbo-configured optimization helpers."""

import logging
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


def prepare_model(architecture) -> None:
    """Prepare VAE layout and lazy compilation before DeepSpeed."""
    from turbo_physai.optimizations.models.openwam.sac import prepare

    prepare(architecture)
    if enabled("checkpoint_skip_layers"):
        if type(architecture).__name__ != "DualSystemSelfAttnArchitecture":
            raise ValueError(
                "checkpoint_skip_layers currently requires dual_system/joint_self_attn"
            )
    layout = enabled("vae_channels_last")
    compile_vae = False
    if enabled("global_compile"):
        from turbo_physai.optimizations.models.openwam.global_compile import (
            compile_scope,
        )

        compile_vae = "vae" in compile_scope()
    if not layout and not compile_vae:
        return
    import torch

    backbone = getattr(architecture, "video_backbone", None)
    vae = getattr(backbone, "vae", None)
    if vae is None:
        logger.warning("[optimizations] VAE preparation skipped: no native Wan VAE")
        return
    if layout:
        torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
        torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    if compile_vae:
        from turbo_physai.optimizations.models.openwam.global_compile import (
            compile_vae_encoder,
        )

        compile_vae_encoder(vae)


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
    vae = getattr(backbone, "vae", None)
    if vae is not None:
        from openwam.model.video_backbone.wan.models.vae import WanVideoVAE
        from turbo_physai.optimizations.models.openwam.ports.vae import (
            WanVideoVAE_prepare_scale_cache,
        )

        if isinstance(vae, WanVideoVAE):
            WanVideoVAE_prepare_scale_cache(vae)


def stage_training_indices(indices, device):
    """Preserve CPU sampling but avoid a synchronizing transfer behind VAE work."""
    import torch

    target = torch.device(device)
    if indices.device.type == "cpu" and target.type == "cuda":
        return indices.pin_memory().to(device=target, non_blocking=True)
    return indices.to(device=target)
