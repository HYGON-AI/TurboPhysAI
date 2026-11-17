"""Skip the unused first-frame VAE conditioning pass for Wan2.2 5B.

"""

from einops import rearrange
import torch
from transformers.feature_extraction_utils import BatchFeature


def encode_image(self, image, num_frames, height, width, encode_y=True):
    with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
        batch_size = image.shape[0]
        clip_context = self.image_encoder.encode_image(image)
        if not encode_y:
            return clip_context, None, None
        image_input = image.transpose(1, 2)
        image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
        self._ensure_vae_on_device(image_input)
        with torch.no_grad():
            y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
        # Build mask to match VAE output shape (VAE may use different spatial downsampling, e.g. WanVideoVAE38 uses patch_size=2 -> height/16)
        # y shape is B * 16 * (1+(T-1)/4) * H_latent * W_latent
        num_t = y.shape[2]
        h_latent, w_latent = y.shape[3], y.shape[4]
        msk = torch.zeros(batch_size, 4, num_t, h_latent, w_latent, dtype=y.dtype, device=self._device)
        msk[:, :, 0:1, :, :] = 1
        new_image = y[:, :, 0:1]
        # concat: B * (4+16) * (1+(T-1)/4) * H_latent * W_latent
        y = torch.concat([msk, y], dim=1)
    return clip_context, y, new_image


def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
    # Set frozen modules to eval
    self.set_frozen_modules_to_eval_mode()

    data = action_input
    # Get embodiment ID.
    embodiment_id = action_input.embodiment_id
    # print("embodiment_id", embodiment_id)
    has_real_action = action_input.has_real_action
    action_mask = action_input.action_mask

    state_features = action_input.state

    actions = action_input.action
    # assert the values of action is in between -1 and 1
    if actions.numel() > 0:
        assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
    videos = data["images"]

    videos = rearrange(videos, "b t h w c -> b c t h w")
    print("videos", videos.shape)


    if videos.dtype == torch.uint8:
        videos = videos.float() / 255.0
        b, c, t, h, w = videos.shape
        videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
        videos = videos.reshape(b * t, c, h, w)
        videos = self.normalize_video(videos)
        videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
        assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
        videos = videos.to(dtype=self.dtype)

    # shape of B * max_length * dim
    prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])

    # Wan 5B: resize to target resolution so latent tokens/frame matches DiT. Use config target when set
    # (e.g. 160x320 so latent is 10x20 with VAE38 16x → even H,W, no crop in dynamics loss); else 176x320.
    target_h = getattr(self.config, "target_video_height", None)
    target_w = getattr(self.config, "target_video_width", None)
    if target_h is None or target_w is None:
        if getattr(self.model, "frame_seqlen", None) in (50, 55):
            target_h, target_w = 176, 320
        else:
            target_h, target_w = None, None
    if target_h is not None and target_w is not None:
        _, _, _, h, w = videos.shape
        if (h, w) != (target_h, target_w):
            b, c, t, _, _ = videos.shape
            videos = torch.nn.functional.interpolate(
                videos.reshape(b * t, c, h, w),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, c, t, target_h, target_w)

    latents = self.encode_video(videos, self.tiled, (self.tile_size_height, self.tile_size_width), (self.tile_stride_height, self.tile_stride_width))

    # print("latents shape", latents.shape, self.dtype)
    _, _, num_frames, height, width = videos.shape
    image = videos[:, :, :1].transpose(1, 2)

    # `ys` (first-frame VAE latent + mask) is only consumed by the DiT when it concatenates it to the input
    # (concat_first_frame_latent) or for model_type 'i2v'. For Wan2.2 5B neither holds, so skip that VAE pass.
    need_ys = getattr(self.model, "concat_first_frame_latent", True) or getattr(self.model, "model_type", None) == "i2v"
    clip_feas, ys, _ = self.encode_image(image, num_frames, height, width, encode_y=need_ys)

    latents = latents.to(self._device)
    clip_feas = clip_feas.to(self._device)
    ys = ys.to(self._device) if ys is not None else None
    prompt_embs = prompt_embs.to(self._device)

    # Loss
    noise = torch.randn_like(latents)

    # specific to autoregressive
    noise = noise.transpose(1, 2)
    latents = latents.transpose(1, 2)

    # ============ VIDEO TIMESTEP SAMPLING ============
    if self.config.decouple_video_action_noise:
        # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
        video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
        timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
        timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
        noise_mode = "DECOUPLED"
    elif self.config.use_high_noise_emphasis:
        # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
        noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
        timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
        timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
        noise_mode = "HIGH_NOISE_EMPHASIS"
    else:
        # Original: uniform sampling over full range
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
        noise_mode = "STANDARD"

    timestep_id_block = timestep_id[:, 1:].reshape(
                timestep_id.shape[0], -1, self.num_frame_per_block)
    timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]

    if actions.numel() > 0:
        noise_action = torch.randn_like(actions)
        assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
        assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"

        # ============ ACTION TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled: sample action timestep independently with full range
            timestep_action_id = torch.randint(
                0,
                self.scheduler.num_train_timesteps,
                (actions.shape[0], actions.shape[1])
            )
            action_mode = "INDEPENDENT"
        else:
            # Original coupled: action timestep derived from video timestep
            timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
            timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
            action_mode = "COUPLED"

        # Log noise mode once
        if not self._noise_logged:
            video_mean = timestep_id.float().mean().item()
            action_mean = timestep_action_id.float().mean().item()
            if noise_mode == "DECOUPLED":
                print(f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}")
            elif noise_mode == "HIGH_NOISE_EMPHASIS":
                print(f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}")
            else:
                print(f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}")
            self._noise_logged = True
    else:
        noise_action = None
        timestep_action_id = None

    timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
    timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
    _, num_frames, num_channels, height, width = noise.shape
    # DiT patch_embedding uses stride (1,2,2), so sequence length is num_frames * (H//2) * (W//2)
    tokens_per_frame = (height // 2) * (width // 2)
    seq_len = num_frames * tokens_per_frame

    timestep = self.scheduler.timesteps[timestep_id].to(self._device)
    noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
    training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)

    if actions.numel() > 0:
        timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
        noisy_actions = self.scheduler.add_noise(
            actions.flatten(0, 1),
            noise_action.flatten(0, 1),
            timestep_action.flatten(0, 1),
        ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
        training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
    else:
        timestep_action = None
        noisy_actions = None
        training_target_action = None

    # Compute loss
    with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
        if actions.numel() > 0:
            video_noise_pred, action_noise_pred = self.model(
                noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                state=state_features, embodiment_id=embodiment_id,
                action=noisy_actions, timestep_action=timestep_action,
                clean_x=latents.transpose(1, 2),
            )
        else:
            video_noise_pred, action_noise_pred = self.model(
                noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action,
                clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                state=state_features, embodiment_id=embodiment_id,
                clean_x=latents.transpose(1, 2),
            )

        # Per-sample dynamics loss
        # DiT patch_embedding uses stride (1,2,2), so output spatial size can be smaller than
        # latent when H or W is odd (e.g. latent 11x20 -> model output 10x20). Crop target to match.
        if training_target.shape != video_noise_pred.shape:
            training_target = training_target[
                ..., : video_noise_pred.shape[3], : video_noise_pred.shape[4]
            ]
        dynamics_loss_per_sample = torch.nn.functional.mse_loss(
            video_noise_pred.float(), training_target.float(), reduction='none'
        ).mean(dim=(1,3,4))  # shape: [B, ...]

        weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
        weighted_dynamics_loss = weight_dynamics.mean()

        if actions.numel() > 0:
            action_loss_per_sample = torch.nn.functional.mse_loss(
                action_noise_pred.float(), training_target_action.float(), reduction='none'
            ) * action_mask  # shape: [B, ...]
            action_loss_per_sample = has_real_action[:, None].float() * action_loss_per_sample  # apply has_real_action
            weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
            weighted_action_loss = weight_action.mean()
            loss = weighted_dynamics_loss + weighted_action_loss
        else:
            weighted_action_loss = torch.tensor(0.0, device=self._device)
            loss = weighted_dynamics_loss
        # loss = dynamics_loss_per_sample.mean()

    # Record log
    output_dict = {
        "loss": loss,
        "dynamics_loss": weighted_dynamics_loss,
        "action_loss": weighted_action_loss,
    }

    return BatchFeature(data=output_dict)
