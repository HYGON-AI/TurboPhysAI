# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from openwam.model.architectures.base import (
    _exclude_vlm_from_state_dict,
    np,
    os,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.architectures.base import (
        Optional,
        Tensor,
    )


def BaseWAMArchitecture__append_proprio_context_token(
    self, pipeline_inputs: dict, proprio: Optional[Tensor]
) -> dict:
    """Append one proprio token to raw text context and extend context_mask.

    If the caller passed a per-sample mask via ``pipeline_inputs['_proprio_sample_mask']``
    (shape (B,) or (B, 1) bool), masked samples get a zero token and a False
    attention mask entry. This isolates proprio_encoder gradients on those
    samples (input * 0 -> weight grad = 0; attention mask cuts the forward path).
    """
    # Always strip the internal routing key so it doesn't leak into downstream forwards,
    # even when proprio context is globally disabled.
    pipeline_inputs = dict(pipeline_inputs)
    from turbo_physai.optimizations.models.openwam.text import trim_text_context

    pipeline_inputs = trim_text_context(pipeline_inputs)
    sample_mask = pipeline_inputs.pop("_proprio_sample_mask", None)
    from turbo_physai.optimizations.models.openwam.attention import cpu_context_mask

    packed_mask = cpu_context_mask(
        pipeline_inputs,
        append_proprio=bool(getattr(self, "_use_proprioception_context", False)),
        sample_mask=sample_mask,
    )

    if not bool(getattr(self, "_use_proprioception_context", False)):
        if packed_mask is not None:
            pipeline_inputs["context_mask"] = packed_mask
        return pipeline_inputs
    if self.proprio_encoder is None:
        raise RuntimeError(
            "proprio context is enabled but proprio_encoder is not initialized."
        )
    if proprio is None:
        raise ValueError(
            "use_proprioception=True requires `proprio` from sample['proprio'] or obs['state']."
        )
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    elif proprio.ndim == 3 and proprio.shape[1] == 1:
        proprio = proprio[:, 0, :]
    if proprio.ndim != 2:
        raise ValueError(
            f"proprio must be [B, D] or [B, 1, D], got shape {tuple(proprio.shape)}"
        )
    if proprio.shape[1] != self.proprio_dim:
        raise ValueError(
            f"proprio last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
        )

    context = pipeline_inputs["context"]
    if context.shape[0] != proprio.shape[0]:
        if proprio.shape[0] == 1 and context.shape[0] > 1:
            proprio = proprio.expand(context.shape[0], -1)
        else:
            raise ValueError(
                f"Batch mismatch between context and proprio: {context.shape[0]} vs {proprio.shape[0]}"
            )

    # Normalize sample_mask to (B, 1) bool on context's device.
    # Accepted incoming shapes (after stacking in _collect_inputs):
    #   * (B,)         — legacy 1D enable-per-sample
    #   * (B, 1)       — legacy "rank-2 enable"
    #   * (B, 1, D)    — 2D per-dim mask emitted by the post-migration
    #                    RoboCOIN/EgoDex/RoboTwin/OXE readers; the
    #                    sample-level enable is ``mask.any(dim=-1)`` so any
    #                    real dim still gates the token in.
    if packed_mask is not None:
        sample_mask = packed_mask[:, -1:]
    elif sample_mask is None:
        sample_mask = torch.ones(
            (proprio.shape[0], 1),
            dtype=torch.bool,
            device=context.device,
        )
    else:
        sample_mask = sample_mask.to(device=context.device, dtype=torch.bool)
        if sample_mask.ndim == 3:
            # (B, 1, D) -> (B, 1): True if any per-dim slot is real
            sample_mask = sample_mask.any(dim=-1)
        elif sample_mask.ndim == 1:
            sample_mask = sample_mask.unsqueeze(-1)
        if sample_mask.shape != (proprio.shape[0], 1):
            raise ValueError(
                f"_proprio_sample_mask shape {tuple(sample_mask.shape)} must be ({proprio.shape[0]}, 1)"
            )

    proprio_token = (
        self.proprio_encoder(
            proprio.to(device=context.device, dtype=self.proprio_encoder.weight.dtype)
        )
        .to(dtype=context.dtype)
        .unsqueeze(1)
    )
    # Token * mask cuts the weight grad for mask=False samples
    # (input goes to zero so the encoder weight gradient contribution is zero).
    sample_mask_f = sample_mask.to(proprio_token.dtype).unsqueeze(-1)  # (B, 1, 1)
    proprio_token = proprio_token * sample_mask_f

    context_mask = pipeline_inputs.get("context_mask")
    if packed_mask is not None:
        context_mask = packed_mask[:, :-1]
    elif context_mask is None:
        seq_lens = pipeline_inputs.get("seq_lens")
        if seq_lens is not None:
            seq_lens = seq_lens.to(device=context.device)
            positions = torch.arange(context.shape[1], device=context.device).unsqueeze(
                0
            )
            context_mask = positions < seq_lens.unsqueeze(1)
        else:
            context_mask = torch.ones(
                (context.shape[0], context.shape[1]),
                dtype=torch.bool,
                device=context.device,
            )
    else:
        context_mask = context_mask.to(device=context.device, dtype=torch.bool)

    updated = dict(pipeline_inputs)
    updated["context"] = torch.cat([context, proprio_token], dim=1)
    updated["context_mask"] = (
        packed_mask
        if packed_mask is not None
        else torch.cat([context_mask, sample_mask], dim=1)
    )
    # The appended proprio token can sit after padded text tokens, so the
    # resulting valid tokens are not necessarily a contiguous prefix.
    # Keep the original text seq_lens and make context_mask authoritative.
    return updated


def BaseWAMArchitecture_save_checkpoint(
    self, path: str, *, state_dict: dict | None = None
) -> None:
    """Save architecture state to safetensors. VLM params excluded (saved separately).

    ``state_dict`` defaults to ``self.state_dict()`` (deploy export); the
    trainer passes a gathered state_dict (ZeRO/DDP all-gather) instead.
    """
    from safetensors.torch import save_file

    if state_dict is None:
        state_dict = self.state_dict()
    state_dict = _exclude_vlm_from_state_dict(state_dict)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Safetensors requires dense contiguous tensors. Channels-last VAE
    # parameters retain their logical shape and values in the checkpoint.
    state_dict = {name: tensor.contiguous() for name, tensor in state_dict.items()}
    save_file(state_dict, path)


@torch.no_grad()
def BaseWAMArchitecture_prepare_inputs(self, batch: list[dict]) -> dict:
    """Aggregate a list of dataset samples into a batched inputs dict.

    The returned dict is designed to be
    unpacked directly into ``compute_loss`` via ``**inputs``.

    Args:
        batch: List of dataset samples (each a dict). A single dict is
            accepted as well and treated as a one-sample batch.

    Returns:
        Dict with all preprocessed video latents, text embeddings, action
        tensors, masks, and forward-time flags ready for ``compute_loss``.
    """
    from openwam.dataloader.transforms.pipeline import FirstFrameConditioningTransform
    from openwam.model.architectures.utils.common import downsample_video_mask_to_latent

    if isinstance(batch, dict):
        batch = [batch]

    if not hasattr(self, "_pipeline_transform_instance"):
        self._pipeline_transform_instance = FirstFrameConditioningTransform()
    samples = [self._pipeline_transform_instance.apply(s) for s in batch]

    _dtype = self.dtype
    _device = self.device

    all_frames: list = []
    all_prompts: list = []
    all_vace_videos: list = []
    all_ref_images: list = []
    all_actions: list = []
    all_proprios: list = []
    all_proprio_masks: list = []
    all_action_masks: list = []
    all_video_masks: list = []

    for sample in samples:
        all_frames.append(sample["video"])
        all_prompts.append(sample["prompt"])
        all_vace_videos.append(sample.get("vace_video"))
        all_ref_images.append(sample.get("first_frame_image"))

        action = sample.get("action")
        if action is not None:
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action)
            action = action.to(dtype=_dtype, device=_device).unsqueeze(0)
        all_actions.append(action)

        # Carry proprio whenever the sample provides it — the main-stream
        # proprio-context path consumes it downstream, so the bridge into
        # ``inputs`` is not gated on a single consumer's flag.
        proprio = sample.get("proprio")
        if proprio is not None:
            if isinstance(proprio, np.ndarray):
                proprio = torch.from_numpy(proprio)
            proprio = proprio.to(dtype=_dtype, device=_device)
            if proprio.ndim == 1:
                pass
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                proprio = proprio[0]
            else:
                raise ValueError(
                    f"sample['proprio'] must be [D] or [1, D], got shape {tuple(proprio.shape)}"
                )
        all_proprios.append(proprio)

        # Collect per-sample proprio_mask. Two accepted shapes:
        #   * 1D ``(1,) bool`` — legacy "is the proprio token enabled".
        #   * 2D ``(1, D) bool`` — per-dim mask; sample-level enable is
        #     ``pmask.any(dim=-1)``. Used by RoboCOIN/EgoDex/RoboTwin/OXE
        #     after the 2D mask migration.
        # Default for readers that don't emit the field: all True (1,).
        # (Mixed 1-D / 2-D ranks across a batch are reconciled just before
        # the stack below, so the bare (1,) default is safe.)
        pmask = sample.get("proprio_mask")
        if pmask is None:
            pmask = torch.ones(1, dtype=torch.bool)
        else:
            if isinstance(pmask, np.ndarray):
                pmask = torch.from_numpy(pmask)
            pmask = pmask.to(dtype=torch.bool)
            if pmask.ndim == 0:
                pmask = pmask.unsqueeze(0)
        all_proprio_masks.append(pmask)

        amask = sample.get("action_mask", None)
        vmask = sample.get("video_mask", None)
        if isinstance(amask, np.ndarray):
            amask = torch.from_numpy(amask)
        if isinstance(vmask, np.ndarray):
            vmask = torch.from_numpy(vmask)
        all_action_masks.append(amask)
        all_video_masks.append(vmask)

    ref_flags = [r is not None for r in all_ref_images]
    if any(ref_flags) and not all(ref_flags):
        raise ValueError(
            "Mixed reference images in batch: all samples must be consistent."
        )

    from turbo_physai.optimizations.models.openwam.latent_cache import stacked

    video_latents = stacked(samples)
    stored = {}
    if video_latents is not None:
        from turbo_physai.optimizations.models.openwam.host_async import upload

        # Pinned non-blocking upload, so this can overlap the step-end collective.
        stored = {
            "video_latents": upload(video_latents, _device),
            "video_size": samples[0]["video_size"],
        }
    preprocessed = self.preprocess(
        frames=all_frames,
        text=all_prompts,
        vace_videos=all_vace_videos,
        ref_images=all_ref_images if ref_flags[0] else None,
        **stored,
    )

    action_data = torch.cat(all_actions, dim=0) if all_actions[0] is not None else None

    inputs = {
        **preprocessed,
        "latents": None,
        "cfg_scale": 1,
        "cfg_merge": False,
        "tiled": False,
        "use_gradient_checkpointing": self._use_gradient_checkpointing,
        "use_gradient_checkpointing_offload": self._use_gradient_checkpointing_offload,
        "max_timestep_boundary": self._max_timestep_boundary,
        "min_timestep_boundary": self._min_timestep_boundary,
        "actions": action_data,
    }

    # Bridge proprio into inputs whenever the batch carries it (not gated on
    # uses_proprioception): the main-stream proprio-context path reads
    # inputs["proprio"]. Consumers that don't need it simply ignore it.
    if all_proprios[0] is not None:
        inputs["proprio"] = torch.stack(all_proprios, dim=0).contiguous()
        # Reconcile mixed 1-D (1,) / 2-D (1, D) proprio_masks before stacking:
        # promote any 1-D enable-flag to (1, D) (broadcasts the sample-level
        # flag across all dims) so a batch mixing a 2-D reader mask with a
        # 1-D default/external mask doesn't raise on rank mismatch. All-1-D
        # and all-2-D batches are left untouched.
        if len({m.ndim for m in all_proprio_masks}) > 1:
            pdim = max(
                (m.shape[-1] for m in all_proprio_masks if m.ndim == 2), default=1
            )
            all_proprio_masks = [
                m if m.ndim == 2 else m.reshape(m.shape[0], 1).expand(m.shape[0], pdim)
                for m in all_proprio_masks
            ]
        inputs["proprio_mask"] = torch.stack(all_proprio_masks, dim=0).contiguous()

    if all_action_masks[0] is not None:
        inputs["action_is_pad"] = torch.stack([~m for m in all_action_masks], dim=0).to(
            device=_device
        )
    if all_video_masks[0] is not None:
        # ``latent[0]`` is a clean conditioning frame (and must be excluded
        # from the loss mask) when either:
        #   (a) the input batch carries ``first_frame_latents`` (Wan TI2V
        #       / cosmos_predict25 TI2V — per-batch signal), in which case
        #       ``base.compute_loss`` will clean-replace ``latents[:, :, 0:1]``
        #       on every step; or
        #   (b) the backbone's *configuration* always reserves ``latent[0]``
        #       for conditioning (only TI2V via the
        #       ``fuse_vae_embedding_in_latents`` / per-token-t=0 path
        #       today).
        # Wan I2V: side-channel ``y`` carries the first-frame reference;
        # ``latent[0]`` itself is fully noised on both train and deploy
        # and must be supervised — NOT in the skip list.
        # Wan VACE: first-frame condition rides on ``vace_context``;
        # video latents are fully noised, ``latent[0]`` enters the loss
        # as a predicted frame. NOT in the skip list.
        # CosmosPredict25 T2V: no first-frame conditioning at all — both
        # signals off.
        skip_first = (
            inputs.get("first_frame_latents") is not None
            or self.video_backbone.needs_first_frame_skip
        )
        # Pass the backbone's temporal_compression so the tail-grouping
        # divisor matches the actual latent-T produced by the encoder.
        # The default 4 in ``downsample_video_mask_to_latent`` is the Wan
        # VAE legacy; for V-JEPA / other encoders it would silently emit
        # a wrong-length mask. See VideoBackbone.temporal_compression for
        # the source-of-truth contract.
        temporal_factor = int(self.video_backbone.temporal_compression)
        latent_masks = [
            downsample_video_mask_to_latent(
                ~m, temporal_factor=temporal_factor, skip_first=skip_first
            )
            for m in all_video_masks
        ]
        inputs["video_is_pad"] = torch.stack(latent_masks, dim=0).to(device=_device)

    return inputs


def BaseWAMArchitecture_compute_loss(
    self,
    *,
    actions: Optional[torch.Tensor] = None,
    lambda_video: float = 1.0,
    lambda_action: float = 1.0,
    **inputs,
) -> dict:
    """Compute joint video-action flow matching loss.

    This is the single entry point for training loss computation.
    Handles timestep sampling, noise injection, forward pass, and
    loss calculation internally.

    Callers should produce ``inputs`` via ``self.prepare_inputs(batch)``
    (preferred) or assemble it manually with the same keys: the output of
    ``self.preprocess()`` plus any of ``actions / proprio /
    action_is_pad / video_is_pad / use_gradient_checkpointing[_offload] /
    max_timestep_boundary / min_timestep_boundary``.

    Args:
        actions: (B, T_action, action_dim) ground truth actions. May also
            be passed via ``inputs["actions"]``.
        lambda_video: Weight for video loss term.
        lambda_action: Weight for action loss term.
        **inputs: Preprocessed video/text tensors plus forward-time flags.

    Returns:
        dict with keys: loss, loss_video, loss_action.
    """
    vb = self.video_backbone
    action_scheduler = self.action_backbone.scheduler
    _dtype = self.dtype
    _device = self.device

    if actions is None:
        actions = inputs.pop("actions", None)
    else:
        inputs.pop("actions", None)

    max_tb = int(inputs.pop("max_timestep_boundary", 1) * len(vb.scheduler.timesteps))
    min_tb = int(inputs.pop("min_timestep_boundary", 0) * len(vb.scheduler.timesteps))
    B = inputs["input_latents"].shape[0]

    # --- Sample video timesteps ---
    video_timestep_ids = torch.randint(min_tb, max_tb, (B,))
    from turbo_physai.optimizations.models.openwam import (
        enabled,
        stage_training_indices,
    )

    if enabled("constant_cache"):
        # Keep the CPU RNG sampling sequence unchanged, then move indices
        # before the forward/backward region instead of at loss weighting.
        # A standalone architecture Group may still have CPU scheduler tables.
        # Index on the table device even when trainer preparation was not run.
        video_timestep_ids = stage_training_indices(
            video_timestep_ids, vb.scheduler.timesteps.device
        )

    video_timesteps = vb.scheduler.timesteps[video_timestep_ids].to(
        dtype=_dtype, device=_device
    )
    video_sigmas = vb.scheduler.sigmas[video_timestep_ids].to(
        dtype=_dtype, device=_device
    )

    # --- Add video noise (flow-matching: linear interp + velocity target) ---
    # Keep seeded noise tied to logical coordinates, independent of VAE layout.
    video_noise = torch.randn_like(
        inputs["input_latents"], memory_format=torch.contiguous_format
    )
    sigma_bc = video_sigmas.view(B, 1, 1, 1, 1)
    inputs["latents"] = (1 - sigma_bc) * inputs[
        "input_latents"
    ] + sigma_bc * video_noise
    video_target = video_noise - inputs["input_latents"]

    if inputs.get("first_frame_latents") is not None:
        inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]

    # --- Prepare action noise ---
    (
        noisy_actions,
        action_target,
        action_timesteps,
        action_timestep_ids,
        action_sigmas,
        a_sigma_bc,
    ) = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if lambda_action > 0 and actions is not None:
        action_timestep_ids = torch.randint(0, len(action_scheduler.timesteps), (B,))
        if enabled("constant_cache"):
            action_timestep_ids = stage_training_indices(
                action_timestep_ids, action_scheduler.timesteps.device
            )

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
        if action_sigmas.dim() == 1:
            a_sigma_bc = action_sigmas.view(B, 1, 1)
        else:
            a_sigma_bc = action_sigmas.unsqueeze(-1)
        noisy_actions = action_scheduler.add_noise(actions, action_noise, a_sigma_bc)
        action_target = action_scheduler.training_target(actions, action_noise)

    # --- Joint forward pass ---
    forward_inputs = dict(inputs)
    proprio = forward_inputs.pop("proprio", None)
    proprio_mask = forward_inputs.pop("proprio_mask", None)
    use_grad_ckpt = forward_inputs.pop("use_gradient_checkpointing", False)
    use_grad_ckpt_offload = forward_inputs.pop(
        "use_gradient_checkpointing_offload", False
    )
    # Padding masks are kept in `inputs` for loss-side masking but dropped
    # from `forward_inputs` so they don't leak into vb.prepare(). Per FastWAM
    # MoT design, attention itself does not consume sample-level padding.
    forward_inputs.pop("action_is_pad", None)
    forward_inputs.pop("video_is_pad", None)

    # Route per-sample proprio_mask through pipeline_inputs to
    # ``_append_proprio_context_token``. Internal-only key; pop'd there.
    if proprio_mask is not None:
        forward_inputs["_proprio_sample_mask"] = proprio_mask

    # Use ``self(...)`` (not ``self.forward(...)``) so ``nn.Module.__call__``
    # is invoked and any architecture-level forward-pre-hooks fire.
    video_noise_pred, action_noise_pred = self(
        noisy_actions if lambda_action > 0 else None,
        action_timesteps if lambda_action > 0 else None,
        proprio=proprio,
        use_gradient_checkpointing=use_grad_ckpt,
        use_gradient_checkpointing_offload=use_grad_ckpt_offload,
        **forward_inputs,
        timestep=video_timesteps,
    )

    # --- Video loss ---
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

    # --- Action loss ---
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

    result = {
        "loss": loss,
        "loss_video": lambda_video * loss_video.detach(),
        "loss_action": lambda_action * loss_action.detach(),
    }

    return result


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

BaseWAMArchitecture__append_proprio_context_token_wrapper = _port_wrapper(
    BaseWAMArchitecture__append_proprio_context_token,
    "openwam.model.architectures.base.BaseWAMArchitecture._append_proprio_context_token",
    "openwam.architecture",
)

BaseWAMArchitecture_save_checkpoint_wrapper = _port_wrapper(
    BaseWAMArchitecture_save_checkpoint,
    "openwam.model.architectures.base.BaseWAMArchitecture.save_checkpoint",
    "openwam.architecture",
)

BaseWAMArchitecture_prepare_inputs_wrapper = _port_wrapper(
    BaseWAMArchitecture_prepare_inputs,
    "openwam.model.architectures.base.BaseWAMArchitecture.prepare_inputs",
    "openwam.architecture",
)

BaseWAMArchitecture_compute_loss_wrapper = _port_wrapper(
    BaseWAMArchitecture_compute_loss,
    "openwam.model.architectures.base.BaseWAMArchitecture.compute_loss",
    "openwam.architecture",
)
