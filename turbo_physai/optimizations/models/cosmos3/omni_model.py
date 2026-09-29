# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""OmniMoTModel adapted-source ports: channels-last conv weights before FSDP
(build_net), non-blocking timestep D2H (training_step), and distributed-optional
set_up_parallelism."""

from __future__ import annotations

import torch

import cosmos_framework.model.generator.omni_mot_model as _omni_mod

from ._source_binding import bind_missing_globals

try:
    from cosmos_framework.utils.memory_format import apply_channels_last_conv_weights
except ModuleNotFoundError as exc:
    if exc.name != "cosmos_framework.utils.memory_format":
        raise
    from ...common.pytorch.memory_format import apply_channels_last_conv_weights

def build_net(self, dtype: torch.dtype, *, lora_enabled: bool | None = None) -> torch.nn.Module:
    # Build model network and parallelize it.
    lora_enabled = self.config.lora_enabled if lora_enabled is None else lora_enabled
    with torch.device("meta"):
        assert self.vlm_config.model_instance is not None, "Model instance should be specified"
        language_model = lazy_instantiate(self.vlm_config.model_instance)

        # NOTE: We pass "RF timesteps" to the network in the same scale as the scheduler
        # (i.e., roughly [0, num_train_timesteps]). The MoT network expects to internally
        # rescale timesteps before embedding; avoid hard-coding 1e-3 by computing it from
        # the configured scheduler resolution.
        num_train_timesteps = self.config.rectified_flow_inference_config.num_train_timesteps
        network_config = Cosmos3VFMNetworkConfig(
            vlm_config=language_model.config,
            latent_patch_size=self.config.diffusion_expert_config.patch_spatial,
            latent_downsample_factor=self.config.latent_downsample_factor,
            latent_channel_size=self.config.state_ch,
            max_latent_h=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
            max_latent_w=self.config.diffusion_expert_config.max_vae_latent_side_after_patchify,
            max_latent_t=self.config.state_t,
            enable_fps_modulation=self.config.diffusion_expert_config.enable_fps_modulation,
            base_fps=self.config.diffusion_expert_config.base_fps,
            vision_gen=self.config.vision_gen,
            action_gen=self.config.action_gen,
            sound_gen=self.config.sound_gen,
            enable_input_bias=self.config.enable_input_bias,
            joint_attn_implementation=self.config.joint_attn_implementation,
            timestep_scale=1.0 / float(num_train_timesteps) * self.config.diffusion_expert_config.timestep_range,
            action_dim=self.config.max_action_dim,
            num_embodiment_domains=self.config.num_embodiment_domains,
            temporal_compression_factor_vision=self.tokenizer_vision_gen.temporal_compression_factor,
            natten_parameter_list=self.config.natten_parameter_list,
            video_temporal_causal=self.config.video_temporal_causal,
            # Sound generation parameters
            sound_dim=self.config.sound_dim,
            sound_latent_fps=self.config.sound_latent_fps,
        )
        network_config._attn_implementation_internal = "eager"
        net = Cosmos3VFMNetwork(
            language_model=language_model,
            config=network_config,
        )
        net.pad_for_cuda_graphs = self.config.compile.use_cuda_graphs

        # Inject LoRA BEFORE FSDP wrap, while still on meta device. The
        # injector must see unsharded Linear shapes; injecting post-FSDP causes
        # lora_B to be created at the per-rank shard size and crashes at
        # forward time. See `OmniMoTModel.add_lora` for details.
        if lora_enabled:
            net = self.add_lora(
                net,
                lora_rank=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_target_modules=self.config.lora_target_modules,
            )

        conv2d_count, conv3d_count = apply_channels_last_conv_weights(net)
        if conv2d_count or conv3d_count:
            log.info(
                "Generator training: enabled channels-last weights before FSDP "
                f"(Conv2d={conv2d_count}, Conv3d={conv3d_count})"
            )

    self.install_attention_dispatch(net)

    net = parallelize_vfm_network(
        net,
        parallel_dims=self.parallel_dims,
        compile_config=self.config.compile,
        ac_config=self.config.activation_checkpointing,
        attention_io_layout=self.config.parallelism.attention_io_layout,
    )

    with misc.timer("meta to cuda and broadcast model states"):
        net = net.to(dtype=dtype)
        net.to_empty(device=DEVICE)
        if DEVICE == Device.CUDA:
            # Weight initialization is not needed for other devices (cpu,
            # meta), since they are only for checkpoint conversion and smoke
            # tests.
            net.init_weights(buffer_device=DEVICE)
            if lora_enabled:
                self._init_lora_weights_post_materialization(net)

    return net


def training_step(
    self, data_batch: dict[str, torch.Tensor], iteration: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """
    Performs a single training step for the rectified-flow (flow-matching) model.

    This method executes one iteration of the model's training. It involves:
    1. Tokenizing generation modalities (vision/action/sound) into latents (tokens).
    2. Sampling a training timestep (t) for each modality and constructing noised latents (xt)
       per the rectified-flow formulation.
    3. Packing text + generation tokens into a single sequence and running the MoT network to predict
       the flow field velocity at the given t.
    4. Computing flow-matching loss (plus optional auxiliary load-balancing losses).

    Args:
        data_batch (dict): raw data batch draw from the training data loader.
        iteration (int): Current iteration number.

    Returns:
        tuple: A tuple containing two elements:
            - dict: additional data that used to debug / logging / callbacks
            - Tensor: The computed loss for the training step as a PyTorch Tensor.

    """
    if self.parallel_dims is None or self.parallel_dims.cp_rank == 0:
        self._update_train_stats(data_batch)

    # Load, apply dropout, and tokenize input captions
    input_text_indexes = self._load_and_tokenize_text_data(data_batch, iteration)

    # Build sequence plans if not present. SequencePlan has the conditioning information.
    sequence_plans = build_sequence_plans_from_data_batch(
        data_batch=data_batch,
        input_video_key=self.input_video_key,
        input_image_key=self.input_image_key,
    )

    # Get data from raw data batch and tokenize into corresponding tokens for *generation* task
    # The unnoised, tokenized data for the generation task.
    gen_data_clean = self.get_data_and_condition(data_batch, iteration=iteration)

    gen_data_clean, memory_info = self.memory_init_training(gen_data_clean, data_batch, input_text_indexes)

    # Compute resolution per sample for per-sample shift lookup
    # image_size[i] may be (1, 4) from IterativeJointDataLoader or (4,) from custom_collate_fn.
    if "image_size" in data_batch:
        data_resolutions = []
        for i in range(gen_data_clean.batch_size):
            img_size = data_batch["image_size"][i]
            if img_size.dim() == 2:
                img_size = img_size[0]
            target_h = int(img_size[0].item())
            target_w = int(img_size[1].item())
            data_resolutions.append(get_vision_data_resolution((target_h, target_w)))
    else:
        data_resolutions = None

    # Calculate number of tokens per sample (before 2x2 merge) for dynamic shift
    # gen_data_clean.x0_tokens_vision: B, C, T, H, W
    assert all(x.shape[0] == 1 for x in gen_data_clean.x0_tokens_vision), (
        "Batch size must be 1 for individual samples"
    )
    num_tokens_per_sample = [x.shape[2] * x.shape[3] * x.shape[4] for x in gen_data_clean.x0_tokens_vision]

    # Sample a random noise level (sigma) and corresponding interpolation coefficient ("timesteps" in RF)
    # Apply shift per sample based on each sample's resolution
    num_vision_latent_frames = [x.shape[2] for x in gen_data_clean.x0_tokens_vision]
    timesteps_vision, sigmas_vision = self._get_train_noise_level_vision(
        batch_size=gen_data_clean.batch_size,
        is_image_batch=gen_data_clean.is_image_batch,
        resolutions=data_resolutions,
        num_vision_latent_frames=num_vision_latent_frames,
        num_tokens=num_tokens_per_sample,
        iteration=iteration,
    )  # [B, T_vis] each

    # Optional independent action schedule (sampled from rectified_flow_action with
    # action-specific shift). Only active when the config opts in and the batch contains
    # action data.
    #
    # Mixed-batch indexing: gen_data_clean.x0_tokens_action (and every packed_sequence.action.*
    # field) is *dense* — one entry per sample with has_action=True, in the original batch order
    # but skipping non-action samples. To feed each dense action entry its sample's sigma, we
    # sample σ for the full batch and reindex with action_sample_indices (the batch positions
    # of action-bearing samples). This avoids the mismatch that happens when, e.g., batch
    # sample 1 has action but the dense entry 0 would otherwise read σ from batch position 0.
    rf_cfg = self.config.rectified_flow_training_config
    action_sample_indices = [i for i, plan in enumerate(sequence_plans) if plan.has_action]
    if rf_cfg.independent_action_schedule and action_sample_indices:
        ts_full, sg_full = self._get_train_noise_level_action(
            batch_size=gen_data_clean.batch_size, iteration=iteration
        )  # [B, 1] each
        idx = torch.tensor(action_sample_indices, dtype=torch.long)  # [n_action]
        timesteps_action = ts_full[idx]  # [n_action, 1]
        sigmas_action = sg_full[idx]  # [n_action, 1]
    else:
        timesteps_action, sigmas_action = (None, None)

    # Optional independent sound schedule: sample a scalar sound sigma per batch
    # slot, then reindex to the dense audio-bearing subset.
    sound_sample_indices = [i for i, plan in enumerate(sequence_plans) if getattr(plan, "has_sound", False)]
    if getattr(rf_cfg, "independent_sound_schedule", False) and sound_sample_indices:
        ts_sound_full, sg_sound_full = self._get_train_noise_level_sound(
            batch_size=gen_data_clean.batch_size
        )  # [B,1] each
        timesteps_sound, sigmas_sound = build_dense_sound_schedule(
            sequence_plans,
            gen_data_clean.x0_tokens_sound,
            ts_sound_full,
            sg_sound_full,
        )  # [n_sound,1], [n_sound,1]
    else:
        timesteps_sound, sigmas_sound = (None, None)

    # Broadcast timesteps/sigmas across CP group to ensure consistency
    if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
        src_rank = 0  # use cp rank 0 to broadcast timesteps/sigmas
        cp_group = self.parallel_dims.cp_mesh.get_group()
        global_src_rank = torch.distributed.get_global_rank(cp_group, src_rank)
        timesteps_vision = timesteps_vision.contiguous()
        sigmas_vision = sigmas_vision.contiguous()
        torch.distributed.broadcast(timesteps_vision, src=global_src_rank, group=cp_group)
        torch.distributed.broadcast(sigmas_vision, src=global_src_rank, group=cp_group)
        if sigmas_action is not None:
            timesteps_action = timesteps_action.contiguous()
            sigmas_action = sigmas_action.contiguous()
            torch.distributed.broadcast(timesteps_action, src=global_src_rank, group=cp_group)
            torch.distributed.broadcast(sigmas_action, src=global_src_rank, group=cp_group)
        if sigmas_sound is not None:
            timesteps_sound = timesteps_sound.contiguous()  # [n_sound,1]
            sigmas_sound = sigmas_sound.contiguous()  # [n_sound,1]
            torch.distributed.broadcast(timesteps_sound, src=global_src_rank, group=cp_group)
            torch.distributed.broadcast(sigmas_sound, src=global_src_rank, group=cp_group)

    if timesteps_sound is None:
        # Sound tensors are dense over audio-bearing samples, while the vision timestep/sigma schedule
        # is indexed by original batch position. Reindex here so mixed audio/no-audio batches use each
        # sound sample's own schedule for noising and loss weighting.
        timesteps_sound, sigmas_sound = build_dense_sound_schedule(
            sequence_plans,
            gen_data_clean.x0_tokens_sound,
            timesteps_vision,
            sigmas_vision,
        )  # [n_sound,T_vis] or None, [n_sound,T_vis] or None

    timesteps_vision_cpu = (
        timesteps_vision.to("cpu", non_blocking=True)
        if timesteps_vision.is_cuda
        else timesteps_vision
    )
    packed_sequence = self._pack_input_sequence(
        sequence_plans,
        input_text_indexes,
        gen_data_clean,
        timesteps_vision_cpu,
        skip_text_tokens=memory_info["skip_text"],
        initial_mrope_temporal_offset=memory_info["initial_temporal_offset"],
    )

    # Under independent_action_schedule, overwrite the vision-based action timestep the
    # packer injected with the action timestep, so the denoiser's action timestep embedding
    # matches the sigma used to noise action tokens.
    if timesteps_action is not None and packed_sequence.action is not None:
        action_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.action.noisy_frame_indexes)
        if action_has_noisy_tokens:
            sample_ts = timesteps_action.squeeze(1).to("cpu", non_blocking=True) if timesteps_action.is_cuda else timesteps_action.squeeze(1)  # [n_action]
            packed_sequence.action.timesteps = torch.cat(
                [
                    sample_ts[i : i + 1].expand(nfi.numel())
                    for i, nfi in enumerate(packed_sequence.action.noisy_frame_indexes)
                ]
            ).to(dtype=torch.float32)  # [N_action_noisy]
        else:
            timesteps_action, sigmas_action = (None, None)

    # Under independent_sound_schedule, overwrite the vision-based sound timestep the packer
    # injected with the sound timestep, so the denoiser's sound timestep embedding matches
    # the sigma used to noise sound tokens.
    if (
        getattr(rf_cfg, "independent_sound_schedule", False)
        and timesteps_sound is not None
        and packed_sequence.sound is not None
    ):
        sound_has_noisy_tokens = any(nfi.numel() > 0 for nfi in packed_sequence.sound.noisy_frame_indexes)
        if sound_has_noisy_tokens:
            sample_ts = timesteps_sound.squeeze(1).to("cpu", non_blocking=True) if timesteps_sound.is_cuda else timesteps_sound.squeeze(1)  # [n_sound]
            packed_sequence.sound.timesteps = torch.cat(
                [
                    sample_ts[i : i + 1].expand(nfi.numel())
                    for i, nfi in enumerate(packed_sequence.sound.noisy_frame_indexes)
                ]
            ).to(dtype=torch.float32)  # [N_sound_noisy]
        else:
            timesteps_sound, sigmas_sound = (None, None)

    # For image editing (multi-item vision), expand per-sample timesteps/sigmas to
    # per-vision-item so downstream noise/loss indexing matches the flat x0_tokens_vision
    # list. No-op when num_vision_items_per_sample is None (standard T2I/T2V/policy cases).
    # Conditioning items get sigma=0 via their condition_mask, so the actual timestep value
    # for them does not matter.
    timesteps_vision = _expand_per_sample_to_per_vision_item(
        timesteps_vision, gen_data_clean.num_vision_items_per_sample
    )  # [B_items, T_vis]
    sigmas_vision = _expand_per_sample_to_per_vision_item(
        sigmas_vision, gen_data_clean.num_vision_items_per_sample
    )  # [B_items, T_vis]

    memory_info = self.pre_noise_memory_hook(packed_sequence, gen_data_clean, memory_info)

    # Flow matching/diffusion forward process: noise the input signal with the sampled noise level
    gen_data_noised = self._add_noise_to_input(
        gen_data_clean,
        packed_sequence,
        sigmas_vision,
        sigmas_action=sigmas_action,
        sigmas_sound=sigmas_sound,
        iteration=iteration,
    )
    self._replace_clean_with_noised(packed_sequence, gen_data_noised)

    # Move packed sequence to CUDA
    packed_sequence.to_cuda()

    # Network forward pass
    memory = self.build_memory_state(packed_sequence, memory_info)  # pylint: disable=assignment-from-none
    out_net = self.denoise(
        data_batch_packed=packed_sequence,
        memory=memory,
    )

    loss, losses_dict = self._compute_losses(
        out_net=out_net,
        data_batch_packed=packed_sequence,
        gen_data_noised=gen_data_noised,
        timesteps=timesteps_vision,
        is_image_batch=gen_data_clean.is_image_batch,
        timesteps_action=timesteps_action,
        timesteps_sound=timesteps_sound,
    )

    # Pixel-space video shapes for VAE FLOPs estimation in callbacks (e.g. MFU).
    _vae_pixel_shapes: list[tuple[int, int, int]] = []
    if gen_data_clean.raw_state_vision is not None:
        for _v in gen_data_clean.raw_state_vision:
            if _v is not None:
                assert _v.dim() in [4, 5], (
                    "Currently only [C, T, H, W] and [B, C, T, H, W] formats are supported for the VAE encoding."
                )
                t_h_w = (
                    (int(_v.shape[2]), int(_v.shape[3]), int(_v.shape[4]))
                    if _v.dim() == 5
                    else (int(_v.shape[1]), int(_v.shape[2]), int(_v.shape[3]))
                )
                _vae_pixel_shapes.append(t_h_w)

    _vision_tokens = len(packed_sequence.vision.sequence_indexes) if packed_sequence.vision else 0
    _action_tokens = len(packed_sequence.action.sequence_indexes) if packed_sequence.action else 0
    _sound_tokens = len(packed_sequence.sound.sequence_indexes) if packed_sequence.sound else 0

    output_batch = {
        "x0": gen_data_clean.x0_tokens_vision,
        "xt": gen_data_noised.xt_tokens_vision,
        "sigma": sigmas_vision,  # [B_items, T_vis]
        "model_pred": out_net["preds_vision"],
        "condition_mask_vision": packed_sequence.vision.condition_mask if packed_sequence.vision else None,
        "condition_mask_action": packed_sequence.action.condition_mask if packed_sequence.action else None,
        "und_token_length": packed_sequence.text_indexes.shape[0],
        "gen_token_length": packed_sequence.sequence_length - packed_sequence.text_indexes.shape[0],
        "vision_token_length": _vision_tokens,
        "action_token_length": _action_tokens,
        "sound_token_length": _sound_tokens,
        "is_image_batch": gen_data_clean.is_image_batch,
        "batch_size": gen_data_clean.batch_size,
        "split_lens": packed_sequence.split_lens,
        "attn_modes": packed_sequence.attn_modes,
        "vae_pixel_shapes": _vae_pixel_shapes,
        **losses_dict,
    }
    if sigmas_action is not None:
        output_batch["sigma_action"] = sigmas_action  # [n_action, 1] — dense over action-bearing samples
    if getattr(rf_cfg, "independent_sound_schedule", False) and sigmas_sound is not None:
        output_batch["sigma_sound"] = sigmas_sound  # [n_sound, 1] — dense over sound-bearing samples

    return output_batch, loss


def set_up_parallelism(self) -> None:
    """Set up the fsdp for the model."""
    if not torch.distributed.is_initialized():
        self.parallel_dims = None
        return

    self.parallel_dims = ParallelDims(
        enable_inference_mode=self.config.parallelism.enable_inference_mode,
        world_size=torch.distributed.get_world_size(),
        dp_shard=self.config.parallelism.data_parallel_shard_degree,
        cfgp=self.config.parallelism.cfg_parallel_shard_degree,
        cp=self.config.parallelism.context_parallel_shard_degree,
    )
    self.parallel_dims.build_meshes(device_type=DEVICE)


bind_missing_globals(globals(), _omni_mod)
