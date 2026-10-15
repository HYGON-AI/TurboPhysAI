# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import types
import torch
import torch.nn as nn
import openwam.model.architectures.dual_system.idm as _source
from turbo_physai.optimizations.common.python.source_binding import bind_missing_globals


def DualSystemIDMArchitecture_compute_loss(
    self,
    *,
    actions: Optional[Tensor] = None,
    lambda_video: float = 1.0,
    lambda_action: float = 1.0,
    **inputs,
) -> dict:
    """IDM training loss with three branches and teacher-forcing mask.

    Branch A: noisy video (denoising target)
    Branch B: noisy action (denoising target)
    Branch C: teacher-forcing cond video (condition for action, optionally noised)

    Branch A/B timesteps are drawn independently with ``torch.randint``.
    """
    vb = self.video_backbone
    ab = self.action_backbone
    action_scheduler = ab.scheduler
    _dtype = self.dtype
    _device = self.device

    if actions is None:
        actions = inputs.pop("actions", None)
    else:
        inputs.pop("actions", None)

    max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
    min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
    input_latents = inputs["input_latents"]
    B = input_latents.shape[0]

    # ---- Branch A: noisy video (denoising target) ----
    video_timestep_ids = torch.randint(min_tb, max_tb, (B,))

    video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(
        dtype=_dtype, device=_device
    )
    video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(
        dtype=_dtype, device=_device
    )

    # Keep seeded noise tied to logical coordinates, independent of VAE layout.
    video_noise = torch.randn_like(input_latents, memory_format=torch.contiguous_format)
    sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
    latents_noisy = (1 - sigma_bc) * input_latents + sigma_bc * video_noise
    video_target = video_noise - input_latents

    if inputs.get("first_frame_latents") is not None:
        latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

    # ---- Branch B: noisy action (denoising target) ----
    noisy_actions = None
    action_target = None
    action_timesteps = None
    action_timestep_ids = None
    if lambda_action > 0 and actions is not None:
        action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))

        action_timesteps = action_scheduler.timesteps[action_timestep_ids].to(
            dtype=_dtype, device=_device
        )
        action_sigmas = action_scheduler.sigmas[action_timestep_ids].to(
            dtype=_dtype, device=_device
        )

        actions = actions.to(dtype=_dtype, device=_device)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)

        action_noise = torch.randn_like(actions)
        a_sigma_bc = action_sigmas.view(B, 1, 1)
        noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
        action_target = action_scheduler.training_target(actions, action_noise)

    # ---- Branch C: teacher-forcing cond video ----
    cond_noise_mask = torch.rand((B,), device=_device) < self.video_cond_noise_prob
    latents_cond = input_latents.clone()

    # Cond video timestep: 0 for clean, sampled for noised
    cond_video_timesteps = torch.zeros((B,), dtype=_dtype, device=_device)
    if bool(cond_noise_mask.any()):
        cond_sampled_ids = torch.randint(min_tb, max_tb, (B,))
        cond_sigmas = vb.scheduler.sigmas[cond_sampled_ids].to(
            dtype=_dtype, device=_device
        )
        cond_sampled_timesteps = vb.scheduler.timesteps[cond_sampled_ids].to(
            dtype=_dtype, device=_device
        )
        cond_video_timesteps = torch.where(
            cond_noise_mask, cond_sampled_timesteps, cond_video_timesteps
        )
        noise_cond = torch.randn_like(
            input_latents, memory_format=torch.contiguous_format
        )
        cond_sigma_bc = cond_sigmas.view(B, 1, 1, 1, 1)
        latents_cond_noisy = (
            1 - cond_sigma_bc
        ) * input_latents + cond_sigma_bc * noise_cond
        selector = cond_noise_mask.view(B, 1, 1, 1, 1)
        latents_cond = torch.where(selector, latents_cond_noisy, latents_cond)

    if inputs.get("first_frame_latents") is not None:
        latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

    # ---- Prepare forward inputs ----
    forward_inputs = dict(inputs)
    proprio = forward_inputs.pop("proprio", None)
    use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
    use_grad_ckpt_offload = forward_inputs.pop(
        "use_gradient_checkpointing_offload", False
    )
    forward_inputs.pop("action_is_pad", None)
    forward_inputs.pop("video_is_pad", None)
    # ``latents`` carries the noisy-branch video latents into the forward;
    # the cond branch is passed alongside as ``cond_video_latents``.
    forward_inputs["latents"] = latents_noisy

    # ---- Run forward through ``self.__call__`` ----
    # Routing through ``self(...)`` (not ``self.forward(...)``) makes
    # ``nn.Module.__call__`` invoke any architecture-level forward-pre-hooks.
    # Mirrors the base ``compute_loss`` pattern.
    video_noise_pred, action_noise_pred = self(
        noisy_actions if lambda_action > 0 else None,
        action_timesteps if lambda_action > 0 else None,
        proprio=proprio,
        use_gradient_checkpointing=use_grad_ckpt,
        use_gradient_checkpointing_offload=use_grad_ckpt_offload,
        cond_video_latents=latents_cond,
        cond_video_timestep=cond_video_timesteps,
        timestep=video_timesteps,
        **forward_inputs,
    )

    # ---- Video loss (same as base) ----
    loss_video = self._compute_video_loss(
        video_noise_pred,
        video_target,
        video_timestep_ids,
        inputs,
        _device,
    )

    if lambda_action == 0 or action_noise_pred is None:
        return {
            "loss": lambda_video * loss_video,
            "loss_video": lambda_video * loss_video.detach(),
            "loss_action": torch.tensor(0.0, device=loss_video.device),
        }

    # ---- Action loss (same as base) ----
    loss_action = self._compute_action_loss(
        action_noise_pred,
        action_target,
        action_timestep_ids,
        action_scheduler,
        inputs,
        _device,
    )

    if lambda_video == 0:
        loss = lambda_action * loss_action
    else:
        loss = lambda_video * loss_video + lambda_action * loss_action

    return {
        "loss": loss,
        "loss_video": lambda_video * loss_video.detach(),
        "loss_action": lambda_action * loss_action.detach(),
    }


bind_missing_globals(globals(), _source)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemIDMArchitecture_compute_loss_wrapper = _port_wrapper(
    DualSystemIDMArchitecture_compute_loss,
    "openwam.model.architectures.dual_system.idm.DualSystemIDMArchitecture.compute_loss",
    "openwam.architecture",
)
