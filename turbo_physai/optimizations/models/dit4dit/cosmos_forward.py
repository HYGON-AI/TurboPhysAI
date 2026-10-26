# Copyright 2025 The NVIDIA Team and The HuggingFace Team.
# Copyright 2025 DiT4DiT team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified for TurboPhysAI; see THIRD_PARTY_NOTICES.md.
# Licensed under the Apache License, Version 2.0.

"""Verified VAE batching and hidden capture; segment compilation is optional."""

from __future__ import annotations

import functools
import os
import types
from typing import Any, List, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from . import video_pack


def _exists(value):
    return value is not None


def _apply_high_sigma_strategy(
    t: torch.Tensor,
    high_sigma_ratio: Optional[float],
    high_sigma_min: Optional[float],
) -> torch.Tensor:
    """Apply high-noise sampling only when the strategy is fully configured."""

    if (
        high_sigma_ratio is None
        or high_sigma_ratio <= 0
        or high_sigma_min is None
    ):
        return t

    high_mask = torch.rand(t.shape, device=t.device) < high_sigma_ratio
    high_t = (
        torch.rand(t.shape, device=t.device, dtype=t.dtype)
        * (1.0 - high_sigma_min)
        + high_sigma_min
    )
    return torch.where(high_mask, high_t, t)


def prepare_latents(
        self,
        *,
        video: Optional[Union[torch.Tensor, List[Any]]],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames_in: int,
        num_frames_out: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        from diffusers.utils.torch_utils import randn_tensor

        B = batch_size
        C = num_channels_latents
        if not _exists(self.vae_scale_factor_temporal) or not _exists(self.vae_scale_factor_spatial):
            raise ValueError("missing VAE scale factors")

        T = (num_frames_out - 1) // int(self.vae_scale_factor_temporal) + 1
        H = height // int(self.vae_scale_factor_spatial)
        W = width // int(self.vae_scale_factor_spatial)
        shape = (B, C, T, H, W)

        if num_frames_in == 0:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            cond_mask = torch.zeros((B, 1, T, H, W), dtype=latents.dtype, device=latents.device)
            cond_indicator = torch.zeros((B, 1, T, 1, 1), dtype=latents.dtype, device=latents.device)
            cond_latents = torch.zeros_like(latents)
            return latents, cond_latents, cond_mask, cond_indicator

        if video is None:
            raise ValueError("`video` must be provided when `num_frames_in` > 0")

        needs_preprocessing = not (
            isinstance(video, torch.Tensor) and video.ndim == 5 and (video.shape[1] == 3 or video.shape[2] == 3)
        )
        if needs_preprocessing:
            raise ValueError("`video` must be a 5D torch.Tensor in (B,3,T,H,W) or (B,T,3,H,W)")

        # Normalize to VAE dtype before encode
        video_t = video.to(device=device, dtype=self.vae.dtype)
        if video_t.shape[1] == 3:
            video_bcthw = video_t
        else:
            # preprocess_video expects (B,T,C,H,W) and returns (B,C,T,H,W)
            video_bcthw = self.video_processor.preprocess_video(video_t, height=height, width=width)

        # Pad video in pixel space to num_frames_out BEFORE VAE encode
        # This matches the official Cosmos pipeline behavior
        temporal_factor = int(self.vae_scale_factor_temporal) if _exists(self.vae_scale_factor_temporal) else 1
        temporal_factor = max(1, temporal_factor)
        T_in_pix = int(video_bcthw.shape[2])

        if T_in_pix < num_frames_out:
            # Pad with zeros in pixel space (matching official implementation)
            n_pad = num_frames_out - T_in_pix
            zero_pad = torch.zeros(
                (video_bcthw.shape[0], video_bcthw.shape[1], n_pad, video_bcthw.shape[3], video_bcthw.shape[4]),
                device=video_bcthw.device,
                dtype=video_bcthw.dtype,
            )
            video_bcthw_padded = torch.cat([video_bcthw, zero_pad], dim=2)
            # video_bcthw_padded = video_bcthw.repeat(1, 1, num_frames_out, 1, 1)
        else:
            video_bcthw_padded = video_bcthw

        # Encode the padded video
        if isinstance(generator, list):
            cond_latents_list = [
                self.vae.encode(video_bcthw_padded[i].unsqueeze(0)).latent_dist.sample(generator=generator[i])
                for i in range(batch_size)
            ]
            cond_latents = torch.cat(cond_latents_list, dim=0)
        else:
            # The VAE supports batched inputs. Encoding each sample separately
            # repeated the complete Conv3D launch sequence once per sample and
            # created an avoidable gap between samples. Keep posterior sampling
            # per sample, though: changing one [B,...] random draw into the old
            # B draws of [1,...] changes the ROCm RNG trajectory and therefore
            # the flow-matching target sequence.
            posterior = self.vae.encode(video_bcthw_padded).latent_dist
            cond_latents_list = []
            for sample_idx in range(batch_size):
                mean_i = posterior.mean[sample_idx : sample_idx + 1]
                std_i = posterior.std[sample_idx : sample_idx + 1]
                noise_i = randn_tensor(
                    mean_i.shape,
                    generator=generator,
                    device=posterior.parameters.device,
                    dtype=posterior.parameters.dtype,
                )
                cond_latents_list.append(mean_i + std_i * noise_i)
            cond_latents = torch.cat(cond_latents_list, dim=0)

        cond_latents = cond_latents.to(dtype)

        latents_mean, latents_std = self._get_latent_stats(
            device=device, dtype=dtype
        )
        cond_latents = (cond_latents - latents_mean) / latents_std

        # Compute output latent shape
        T_out_lat = (num_frames_out - 1) // temporal_factor + 1

        H_lat = int(cond_latents.shape[-2])
        W_lat = int(cond_latents.shape[-1])
        shape_out = (B, C, int(T_out_lat), H_lat, W_lat)

        if latents is None:
            # Generate noise with same pattern across batch dimension
            if generator is not None:
                single_noise = randn_tensor((1, C, int(T_out_lat), H_lat, W_lat), generator=generator, device=device, dtype=dtype)
                latents = single_noise.repeat(B, 1, 1, 1, 1)
            else:
                latents = randn_tensor(shape_out, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if tuple(latents.shape) != shape_out:
                raise ValueError(f"Provided `latents` shape {tuple(latents.shape)} != expected latent shape {shape_out}")

        # cond_latents should now already have the correct shape from VAE encode
        # Just ensure it matches shape_out
        if cond_latents.shape[2] != T_out_lat:
            # Truncate or pad if needed (should rarely happen)
            cond_latents_adjusted = latents.new_zeros(shape_out)
            T_copy = min(int(cond_latents.shape[2]), int(T_out_lat))
            cond_latents_adjusted[:, :, :T_copy] = cond_latents[:, :, :T_copy].to(dtype=dtype, device=device)
            cond_latents = cond_latents_adjusted

        _, _, T_lat, H_lat, W_lat = shape_out
        num_cond_latent_frames = min(T_lat, (num_frames_in - 1) // temporal_factor + 1)
        cond_mask, cond_indicator = self._get_condition_tensors(
            batch_size=B,
            num_frames=T_lat,
            height=H_lat,
            width=W_lat,
            num_condition_frames=num_cond_latent_frames,
            device=latents.device,
            dtype=latents.dtype,
        )

        return latents, cond_latents, cond_mask, cond_indicator

def feature_extractor_forward(
        self,
        *,
        prompts: Union[str, List[str]],
        videos: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int = 1,
        capture_step_index: int = 0,
        conditional_frame_timestep: float = 0.001,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        detach: bool = True,
        num_frames_out: Optional[int] = None,
        gt_future_videos: Optional[torch.Tensor] = None,  # (B,Tf,3,H,W) float in [0,1]
        return_pred_future_video: bool = False,
        fixed_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16 but are {height} and {width}.")

        device = self.device
        transformer_dtype = self.transformer.dtype

        b = int(videos.shape[0])
        if isinstance(prompts, str):
            prompts_list: List[str] = [prompts] * b
        else:
            prompts_list = prompts
            if len(prompts_list) != b:
                raise ValueError(f"len(prompts)={len(prompts_list)} must match batch size B={b}")

        # Create fixed generator if seed is provided
        if fixed_seed is not None and generator is None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(fixed_seed)

        # ctx = torch.enable_grad() if self.trainable else torch.no_grad()
        with torch.enable_grad():
            prompt_embeds, _ = self.encode_prompt(
                prompt=prompts_list,
                guidance_scale=1.0,
                num_videos_per_prompt=1,
                device=device,
                max_sequence_length=self.max_sequence_length,
            )

            videos_bcthw = self._coerce_videos_to_bcthw(videos, height=height, width=width)
            videos_bcthw = videos_bcthw.to(device=device, dtype=self.vae.dtype) ##[-1,1]
            num_frames_in = int(videos_bcthw.shape[2])
            # NOTE: Cosmos VAE is temporally downsampled (vae_scale_factor_temporal > 1).
            temporal_factor = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
            temporal_factor = max(1, temporal_factor)
            min_frames_out_for_future = 1 + temporal_factor

            if num_frames_out is None:
                if gt_future_videos is not None and torch.is_tensor(gt_future_videos):
                    # gt_future_videos: (B,T_f,3,H,W) or (B,3,T_f,H,W)
                    gt_t_dim = gt_future_videos.shape[1] if gt_future_videos.shape[2] == 3 else gt_future_videos.shape[2]
                    num_frames_out = num_frames_in + int(gt_t_dim)
                else:
                    num_frames_out = num_frames_in
                # Ensure enough output frames for at least one future latent slot
                num_frames_out = max(int(num_frames_out), int(min_frames_out_for_future))

            # If we need any future prediction/supervision, ensure enough output frames to create a future latent slot.
            # if (gt_future_videos is not None and torch.is_tensor(gt_future_videos)) or bool(return_pred_future_video):
            #     num_frames_out = max(int(num_frames_out), int(min_frames_out_for_future))

            num_channels_latents = int(self.transformer.config.in_channels) - 1
            latents, cond_latents, cond_mask, cond_indicator = self.prepare_latents(
                video=videos_bcthw,
                batch_size=b,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames_in=num_frames_in,
                num_frames_out=int(num_frames_out),
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=None,
            )

            # This is determined entirely by static shapes and Python integers.
            # Keep it on the host instead of reducing cond_indicator on the GPU
            # and calling .item(), which synchronizes the device every forward.
            cond_count = min(
                int(latents.shape[2]),
                (num_frames_in - 1) // temporal_factor + 1,
            )

            (
                cond_timestep,
                cond_mask_t,
                inv_cond_mask_t,
                inv_cond_indicator,
                padding_mask,
            ) = self._get_forward_static_tensors(
                condition_frames=cond_count,
                cond_mask=cond_mask,
                cond_indicator=cond_indicator,
                transformer_dtype=transformer_dtype,
                conditional_frame_timestep=conditional_frame_timestep,
                height=height,
                width=width,
            )

            if not self._direct_hidden_capture:
                self._cached_hidden.clear()
                self._capture_hidden_enabled = True

            # Resolve future loss type early so we can skip expensive sampling/decoding when training with
            # latent-space flow-matching loss only.
            loss_type = None
            if gt_future_videos is not None:
                cosmos_cfg = getattr(getattr(self, "config", None), "framework", None)
                cosmos_cfg = getattr(cosmos_cfg, "cosmos25", None) if cosmos_cfg is not None else None
                if cosmos_cfg is not None:
                    if isinstance(cosmos_cfg, dict):
                        loss_type = str(cosmos_cfg.get("future_loss_type", "pixel_l1")).lower()
                    else:
                        loss_type = str(getattr(cosmos_cfg, "future_loss_type", "pixel_l1")).lower()
                else:
                    loss_type = "pixel_l1"

            flow_matching_types = ("flow_matching", "latent_flow_matching", "rectified_flow", "rf")
            train_flow_matching_only = (gt_future_videos is not None) and (loss_type in flow_matching_types) and (not return_pred_future_video)

            # If we need generated video / future supervision, run the full denoising loop.
            # Otherwise, we can early-exit after `capture_step_index` (default 0).
            #
            # Match diffusers' Cosmos2.5 Predict pipeline behavior:
            # - Use UniPCMultistepScheduler with `prediction_type="flow_prediction"` + `use_flow_sigmas=True`
            # - Treat transformer output as velocity (a.k.a. flow) and replace conditioned frames with gt_velocity
            # For pure flow-matching training, we don't need to run the sampling loop or decode videos.
            need_generate = ((gt_future_videos is not None) or bool(return_pred_future_video)) and (not train_flow_matching_only)

            # If we aren't generating, only run enough steps to reach capture_step_index (usually 0).
            steps_for_hidden = int(num_inference_steps) if need_generate else max(1, int(capture_step_index) + 1)
            self.scheduler.set_timesteps(steps_for_hidden, device=device)
            timesteps = self.scheduler.timesteps

            stop_i = (len(timesteps) - 1) if need_generate else capture_step_index
            if stop_i < 0 or stop_i >= len(timesteps):
                raise ValueError(f"stop_i={stop_i} out of range for {len(timesteps)} steps")

            # Determine whether scheduler expects flow/velocity outputs.
            scheduler_cfg = getattr(self.scheduler, "config", None)
            prediction_type = str(getattr(scheduler_cfg, "prediction_type", "")).lower() if scheduler_cfg is not None else ""
            use_flow_prediction = prediction_type == "flow_prediction"

            # In diffusers pipeline_cosmos2_5_predict.py:
            #   gt_velocity = (latents - cond_latent) * cond_mask
            # where `latents` are the initial noise latents and `cond_latent` are the (padded) conditional latents.
            gt_velocity = None
            if use_flow_prediction:
                gt_velocity = (latents - cond_latents).to(dtype=transformer_dtype) * cond_mask_t

            hidden_first: Optional[torch.Tensor] = None
            for i, t in enumerate(timesteps):
                # NOTE: for Cosmos2.5 Predict with flow sigmas, scheduler.sigmas are mapped to [0,1]
                # Avoid the GPU -> CPU -> GPU round trip caused by .item().
                sigma_t = torch.as_tensor(
                    self.scheduler.sigmas[i],
                    device=device,
                    dtype=transformer_dtype,
                ).reshape(1)
                in_latents = cond_mask_t * cond_latents + inv_cond_mask_t * latents
                in_latents = in_latents.to(transformer_dtype)
                in_timestep = cond_indicator * cond_timestep + inv_cond_indicator * sigma_t

                capture_only = bool(
                    self._direct_hidden_capture and i == 0 and not need_generate
                )
                transformer_outputs = self.transformer(
                    hidden_states=in_latents,
                    condition_mask=cond_mask_t,
                    timestep=in_timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                    **(
                        {
                            "capture_layer": self.extract_layer
                            if i == 0 else None,
                            "stop_after_capture": capture_only,
                        }
                        if self._direct_hidden_capture
                        else {}
                    ),
                )
                model_out = transformer_outputs[0]

                # Replace conditional-frame velocity with gt_velocity (diffusers behavior)
                if model_out is not None and gt_velocity is not None:
                    model_out = gt_velocity + model_out * inv_cond_mask_t


                # Capture hidden from the first transformer call. The optimized
                # path returns it explicitly; the fallback keeps hook behavior.
                if i == 0 and hidden_first is None:
                    if self._direct_hidden_capture:
                        captured_hidden = transformer_outputs[1]
                        if captured_hidden is None:
                            raise RuntimeError(
                                "Segmented Cosmos forward did not return "
                                f"capture_layer={self.extract_layer}"
                            )
                        hidden_first = captured_hidden.detach()
                    elif len(self._cached_hidden) > 0:
                        hidden_first = self._cached_hidden[-1]
                        self._capture_hidden_enabled = False

                # Only advance diffusion if we actually need generated video / future loss.
                if need_generate:
                    if model_out is None:
                        raise RuntimeError(
                            "Transformer prediction is required for generation"
                        )
                    latents = self.scheduler.step(model_out, t, latents, return_dict=False)[0]

                if i == stop_i:
                    break

            if hidden_first is None and len(self._cached_hidden) == 0:
                raise RuntimeError(
                    "No transformer hidden captured. Your transformer may not expose `transformer_blocks`, "
                    "or `extract_layer` is not a valid block index."
                )

            hidden = hidden_first if hidden_first is not None else self._cached_hidden[-1]

            # Optional future-video prediction + auxiliary supervision loss
            future_loss = None
            pred_video = None
            # gt_future_videos = None
            if gt_future_videos is not None or return_pred_future_video:
                # Only decode when needed:
                # - return_pred_future_video=True
                # - pixel-space supervision (pixel_l1 / pixel_mse)
                need_decode = bool(return_pred_future_video) or (
                    (gt_future_videos is not None) and (loss_type is not None) and (loss_type not in ("latent_mse",) + flow_matching_types)
                )
                pred_video_full = None
                if need_decode:
                    pred_video_full = self._decode_latents_to_video(latents)  # (B,T,3,H,W)
                    if return_pred_future_video:
                        pred_video = pred_video_full

                if gt_future_videos is not None:
                    gt = gt_future_videos
                    if gt.ndim == 5 and gt.shape[2] == 3:
                        # (B,T,3,H,W) OK
                        pass
                    elif gt.ndim == 5 and gt.shape[1] == 3:
                        # (B,3,T,H,W) -> (B,T,3,H,W)
                        # Keep this as a zero-copy view.  The flow/latent paths
                        # below transpose it back to B,C,T,H,W before cat, and
                        # pixel losses accept strided tensors directly.
                        gt = gt.permute(0, 2, 1, 3, 4)
                    else:
                        raise ValueError(f"gt_future_videos must be (B,T,3,H,W) or (B,3,T,H,W), got {tuple(gt.shape)}")

                    # Loss type: pixel_l1 | pixel_mse | latent_mse | flow_matching (latent-space)
                    # `loss_type` is resolved above to allow skipping sampling/decode in flow-matching-only training.
                    loss_type = loss_type or "pixel_l1"

                    if loss_type in flow_matching_types:
                        # Latent-space flow matching (rectified flow) loss, only on future latent frames.
                        #
                        # We construct x_t = (1 - t) * x0 + t * z, and train the model to predict velocity:
                        #   v*(x_t, t) = z - x0
                        # conditioned frames are masked out, and we only supervise future frames.

                        # Build full GT pixel video (B,3,T_in+T_f,H,W) and pad GT if needed so that
                        # VAE temporal downsampling produces at least one "future latent slot" (important when tf>1).
                        # The pack function accepts this strided view and writes
                        # directly into its final output, so an intermediate
                        # contiguous copy would only add device traffic.
                        gt_bcthw = gt.permute(0, 2, 1, 3, 4)
                        # gt 来自 dataloader，契约为 [0,1]。直接转换可避免
                        # gt_bcthw.min() 的 GPU reduction 和 Python bool 同步。
                        full_bcthw = video_pack.pack_flow_matching_video(
                            videos_bcthw,
                            gt_bcthw,
                        )

                        # Ensure we have enough *pixel* frames to create a future latent timestep.
                        temporal_factor = int(getattr(self, "vae_scale_factor_temporal", 1) or 1)
                        temporal_factor = max(1, temporal_factor)
                        min_full_frames = 1 + temporal_factor  # 2 latent timesteps in pixel domain
                        if int(full_bcthw.shape[2]) < int(min_full_frames):
                            n_pad = int(min_full_frames - int(full_bcthw.shape[2]))
                            last = full_bcthw[:, :, -1:, :, :].repeat(1, 1, n_pad, 1, 1)
                            full_bcthw = torch.cat([full_bcthw, last], dim=2)

                        with torch.no_grad():
                            gt_latents_norm = self._encode_video_to_latents_norm(full_bcthw)

                        pred_latents_future = latents[:, :, cond_count:]
                        gt_latents_future = gt_latents_norm[:, :, cond_count : cond_count + pred_latents_future.shape[2]]

                        # If we still don't have a future latent to supervise, return 0.
                        if pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            # Align lengths (just in case)
                            T_sup = min(int(pred_latents_future.shape[2]), int(gt_latents_future.shape[2]))
                            x0_future = gt_latents_future[:, :, :T_sup].to(device=latents.device, dtype=torch.float32)

                            B_fm = latents.shape[0]
                            # Read flow-matching config: time_distribution and high_sigma_strategy
                            fm_cfg = getattr(cosmos_cfg, "flow_matching", None) if cosmos_cfg is not None else None
                            time_dist = str(getattr(fm_cfg, "time_distribution", "logit_normal")) if fm_cfg else "logit_normal"
                            # high_sigma_ratio = float(getattr(fm_cfg, "high_sigma_ratio", 0.05)) if fm_cfg else 0.05
                            _hsr = getattr(fm_cfg, "high_sigma_ratio", 0.05) if fm_cfg else 0.05
                            _hsm = getattr(fm_cfg, "high_sigma_min", 0.98) if fm_cfg else 0.98
                            high_sigma_ratio = None if _hsr is None else float(_hsr)
                            high_sigma_min = None if _hsm is None else float(_hsm)

                            # Sample rectified-flow time t in [0,1]
                            if time_dist == "logit_normal":
                                # Logit-normal: t = sigmoid(N(0,1)), concentrates around t≈0.5
                                t = torch.sigmoid(torch.randn((B_fm,), device=latents.device, dtype=torch.float32))
                            else:
                                t = torch.rand((B_fm,), device=latents.device, dtype=torch.float32)

                            # High sigma strategy: force a fraction of samples to high noise (t close to 1)
                            t = _apply_high_sigma_strategy(
                                t,
                                high_sigma_ratio,
                                high_sigma_min,
                            )

                            t = t.view(B_fm, 1, 1, 1, 1)  # (B,1,1,1,1) for latent broadcast

                            # Sample noise z
                            z_future = torch.randn_like(x0_future)

                            # x_t for future frames only
                            xt_future = (1.0 - t) * x0_future + t * z_future

                            # Build a full latent tensor for conditioning:
                            # - cond frames: pinned via in_latents mixing (cond_mask_t * cond_latents)
                            # - future frames: use x_t for supervised portion; fill the rest with noise
                            xt_full = torch.randn_like(latents[:, :, :, :, :].float())
                            xt_full[:, :, cond_count : cond_count + T_sup] = xt_future

                            # timestep tensor shaped like (B,1,T,1,1): cond frames use cond_timestep, future use t
                            t_B1T11 = latents.new_zeros(cond_indicator.shape, dtype=torch.float32)
                            t_B1T11[:, :, cond_count : cond_count + T_sup] = t  # broadcast over time slice
                            t_B1T11 = t_B1T11.to(dtype=transformer_dtype)

                            in_latents_fm = cond_mask_t * cond_latents + inv_cond_mask_t * xt_full.to(transformer_dtype)
                            in_timestep_fm = cond_indicator * cond_timestep + inv_cond_indicator * t_B1T11

                            v_pred = self.transformer(
                                hidden_states=in_latents_fm,
                                condition_mask=cond_mask_t,
                                timestep=in_timestep_fm,
                                encoder_hidden_states=prompt_embeds,
                                padding_mask=padding_mask,
                                return_dict=False,
                            )[0]

                            # Supervise only future frames: v* = z - x0
                            v_tgt_future = (z_future - x0_future).to(device=v_pred.device, dtype=v_pred.dtype)
                            v_pred_future = v_pred[:, :, cond_count : cond_count + T_sup]

                            future_loss = F.mse_loss(v_pred_future.float(), v_tgt_future.float())

                    elif loss_type == "latent_mse":
                        # Build full GT pixel video (B,3,T_in+T_f,H,W)
                        # torch.cat supports strided inputs and writes a new
                        # contiguous output; avoid materializing gt twice.
                        gt_bcthw = gt.permute(0, 2, 1, 3, 4)
                        full_bcthw = torch.cat([videos_bcthw.to(gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2)

                        with torch.no_grad():
                            gt_latents_norm = self._encode_video_to_latents_norm(full_bcthw)

                        pred_latents_future = latents[:, :, cond_count:]
                        gt_latents_future = gt_latents_norm[:, :, cond_count : cond_count + pred_latents_future.shape[2]]

                        if pred_latents_future.numel() == 0 or gt_latents_future.numel() == 0:
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            future_loss = F.mse_loss(
                                pred_latents_future.float(),
                                gt_latents_future.to(pred_latents_future.device, dtype=torch.float32),
                            )
                    else:
                        # Pixel supervision: only on future frames (skip t0)
                        if pred_video_full is None:
                            # Should not happen because pixel loss requires decode, but keep safe.
                            future_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
                        else:
                            tf = int(gt.shape[1])
                            if pred_video_full.shape[1] >= 1 + tf:
                                pred_future = pred_video_full[:, 1 : 1 + tf]
                            else:
                                pred_future = pred_video_full[:, -tf:]
                            gt_pix = gt.to(pred_future.device, dtype=pred_future.dtype)
                            if loss_type in ("pixel_mse", "mse"):
                                future_loss = F.mse_loss(pred_future, gt_pix)
                            else:
                                future_loss = torch.abs(pred_future - gt_pix).mean()

            if detach:
                hidden = hidden.detach()
                if pred_video is not None:
                    pred_video = pred_video.detach()

            if gt_future_videos is not None or return_pred_future_video:
                return hidden, pred_video, future_loss  # type: ignore[return-value]

            return hidden

# The segmented transformer implementation below derives from Diffusers.
# Copyright 2025 The NVIDIA Team and The HuggingFace Team.
# Licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
def _make_cosmos_segment(
    transformer: nn.Module,
    start: int,
    end: int,
    compile_kwargs: dict,
):
    """Compile a consecutive block range without changing module ownership."""

    def run_segment(
        hidden_states: torch.Tensor,
        processed_encoder_hidden_states,
        embedded_timestep: torch.Tensor,
        temb: torch.Tensor | None,
        image_rotary_emb,
        extra_pos_emb: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # Resolve blocks from the original ModuleList so parameter names and
        # state_dict layout remain unchanged.
        for block_idx in range(start, end):
            hidden_states = transformer.transformer_blocks[block_idx](
                hidden_states,
                processed_encoder_hidden_states,
                embedded_timestep,
                temb,
                image_rotary_emb,
                extra_pos_emb,
                attention_mask,
                None,
            )
        return hidden_states

    return run_segment if compile_kwargs is None else torch.compile(run_segment, **compile_kwargs)


def _get_expanded_padding_mask(
    transformer,
    padding_mask: torch.Tensor,
    *,
    spatial_size: tuple[int, int],
    batch_size: int,
    num_frames: int,
) -> torch.Tensor:
    """Resize and expand one mask without unsafe allocator-address reuse."""
    padding_cache = transformer._dit4dit_padding_mask_cache
    padding_key = (
        id(padding_mask),
        padding_mask._version,
        tuple(padding_mask.shape),
        tuple(spatial_size),
        batch_size,
        num_frames,
        str(padding_mask.device),
        padding_mask.dtype,
    )
    cached = padding_cache.get(padding_key)
    if cached is not None:
        cached_mask, cached_expanded = cached
        if cached_mask is padding_mask:
            return cached_expanded

    padding_mask_resized = transforms.functional.resize(
        padding_mask,
        list(spatial_size),
        interpolation=transforms.InterpolationMode.NEAREST,
    )
    padding_mask_expanded = padding_mask_resized.unsqueeze(2).repeat(
        batch_size, 1, num_frames, 1, 1
    )

    # Retain the source tensor in each entry. This prevents Python object IDs
    # and allocator addresses from being reused while the cached result is
    # reachable. A version change invalidates an in-place modified tensor.
    for stale_key, (cached_mask, _) in tuple(padding_cache.items()):
        if cached_mask is padding_mask:
            del padding_cache[stale_key]
    if len(padding_cache) >= 4:
        padding_cache.clear()
    padding_cache[padding_key] = (padding_mask, padding_mask_expanded)
    return padding_mask_expanded


def _segmented_cosmos_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    block_controlnet_hidden_states: list[torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    fps: int | None = None,
    condition_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    return_dict: bool = True,
    capture_layer: int | None = None,
    stop_after_capture: bool = False,
):
    """Cosmos forward with compiled block segments and explicit capture.

    ``stop_after_capture`` is used by the hidden-only feature-extraction pass.
    It avoids executing the remaining transformer blocks and output head when
    the caller consumes only the captured intermediate hidden state.
    """
    if block_controlnet_hidden_states is not None:
        raise NotImplementedError(
            "Segmented Cosmos forward does not support ControlNet residuals"
        )
    if stop_after_capture and (capture_layer is None or return_dict):
        raise ValueError(
            "stop_after_capture requires capture_layer and return_dict=False"
        )

    batch_size, _, num_frames, height, width = hidden_states.shape

    if condition_mask is not None:
        hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

    if self.config.concat_padding_mask:
        padding_mask_expanded = _get_expanded_padding_mask(
            self,
            padding_mask,
            spatial_size=tuple(hidden_states.shape[-2:]),
            batch_size=batch_size,
            num_frames=num_frames,
        )
        hidden_states = torch.cat(
            [hidden_states, padding_mask_expanded],
            dim=1,
        )

    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    image_rotary_emb = self.rope(hidden_states, fps=fps)
    extra_pos_emb = (
        self.learnable_pos_embed(hidden_states)
        if self.config.extra_pos_embed_type
        else None
    )

    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    hidden_states = self.patch_embed(hidden_states)
    hidden_states = hidden_states.flatten(1, 3)

    if timestep.ndim == 1:
        temb, embedded_timestep = self.time_embed(hidden_states, timestep)
    elif timestep.ndim == 5:
        if timestep.shape != (batch_size, 1, num_frames, 1, 1):
            raise ValueError(
                "Expected timestep shape "
                f"{(batch_size, 1, num_frames, 1, 1)}, got {timestep.shape}"
            )
        timestep = timestep.flatten()
        temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        temb, embedded_timestep = (
            value.view(batch_size, post_patch_num_frames, 1, 1, -1)
            .expand(-1, -1, post_patch_height, post_patch_width, -1)
            .flatten(1, 3)
            for value in (temb, embedded_timestep)
        )
    else:
        raise ValueError(
            "Expected timestep to have shape [B, 1, T, 1, 1] or [T], "
            f"got {timestep.shape}"
        )

    text_context, img_context = (
        encoder_hidden_states
        if isinstance(encoder_hidden_states, tuple)
        else (encoder_hidden_states, None)
    )
    if self.config.use_crossattn_projection:
        text_context = self.crossattn_proj(text_context)
    if img_context is not None and self.config.img_context_dim_in:
        img_context = self.img_context_proj(img_context)
    processed_encoder_hidden_states = (
        (text_context, img_context)
        if isinstance(encoder_hidden_states, tuple)
        else text_context
    )

    captured_hidden = None
    segment_ranges = self._dit4dit_segment_ranges
    segments = self._dit4dit_compiled_segments
    for (start, end), segment in zip(segment_ranges, segments):
        segment_args = (
            hidden_states,
            processed_encoder_hidden_states,
            embedded_timestep,
            temb,
            image_rotary_emb,
            extra_pos_emb,
            attention_mask,
        )
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                segment, *segment_args
            )
        else:
            hidden_states = segment(*segment_args)

        if capture_layer is not None and end - 1 == capture_layer:
            captured_hidden = hidden_states
            if stop_after_capture:
                break

    if capture_layer is not None and captured_hidden is None:
        raise RuntimeError(
            f"capture_layer={capture_layer} is not a segment boundary: "
            f"{segment_ranges}"
        )

    if stop_after_capture:
        # The first item intentionally has no model prediction. The caller
        # requested this pass solely for the intermediate feature tensor.
        return None, captured_hidden

    hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
    hidden_states = hidden_states.unflatten(
        1, (post_patch_num_frames, post_patch_height, post_patch_width)
    )
    hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return hidden_states, captured_hidden
    return Transformer2DModelOutput(sample=hidden_states)


def install_segmented_cosmos_forward(
    transformer: nn.Module,
    *,
    capture_layer: int,
    segment_size: int = 6,
    compile_kwargs: dict | None = None,
) -> tuple[tuple[int, int], ...]:
    """Compile block ranges and replace the block-by-block Python loop."""
    if getattr(transformer, "_dit4dit_segmented_forward", False):
        return transformer._dit4dit_segment_ranges
    if transformer.__class__.__name__ != "CosmosTransformer3DModel":
        raise TypeError(
            "Expected CosmosTransformer3DModel, got "
            f"{transformer.__class__.__name__}"
        )
    if segment_size < 1:
        raise ValueError("segment_size must be >= 1")

    block_count = len(transformer.transformer_blocks)
    if capture_layer < 0 or capture_layer >= block_count:
        raise ValueError(
            f"capture_layer={capture_layer} is outside [0, {block_count})"
        )

    boundaries = set(range(0, block_count + 1, segment_size))
    boundaries.update((0, capture_layer + 1, block_count))
    ordered = sorted(boundaries)
    segment_ranges = tuple(zip(ordered[:-1], ordered[1:]))
    kwargs = None if compile_kwargs is None else dict(compile_kwargs)
    compiled_segments = tuple(
        _make_cosmos_segment(transformer, start, end, kwargs)
        for start, end in segment_ranges
    )

    # These are runtime callables, not child modules. Keep them out of
    # state_dict so existing checkpoints retain exactly the same key layout.
    object.__setattr__(transformer, "_dit4dit_segment_ranges", segment_ranges)
    object.__setattr__(
        transformer, "_dit4dit_compiled_segments", compiled_segments
    )
    object.__setattr__(transformer, "_dit4dit_padding_mask_cache", {})
    object.__setattr__(transformer, "_dit4dit_segmented_forward", True)
    transformer.forward = types.MethodType(_segmented_cosmos_forward, transformer)
    return segment_ranges

def register_direct_capture(self):
    install_segmented_cosmos_forward(
        self.transformer, capture_layer=int(self.extract_layer),
        segment_size=len(self.transformer.transformer_blocks),
    )
    self._direct_hidden_capture = True


def compile_segment_wrapper(original, options):
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    from torch._inductor import config

    size = int(options.get("segment_size", 6))
    if size < 1:
        raise ValueError("segment_size must be >= 1")
    if not hasattr(config, "alignment_asserts"):
        raise RuntimeError("This PyTorch build does not support alignment_asserts")
    if options.get("alignment_asserts", False) is not False:
        raise ValueError("Cosmos segment compile requires alignment_asserts=False")
    if options.get("mode", "default") != "default":
        raise ValueError("Cosmos alignment workaround requires mode=default")
    kwargs = dict(
        backend="inductor",
        fullgraph=bool(options.get("fullgraph", False)),
        dynamic=bool(options.get("dynamic", False)),
        options={"alignment_asserts": False},
    )

    @functools.wraps(original)
    def wrapped(transformer, *, capture_layer, segment_size=6, compile_kwargs=None):
        return original(
            transformer, capture_layer=capture_layer,
            segment_size=size, compile_kwargs=kwargs,
        )
    return wrapped
