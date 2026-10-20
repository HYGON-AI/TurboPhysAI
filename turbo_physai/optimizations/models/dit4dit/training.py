# Copyright 2025 DiT4DiT team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Adapted from DiT4DiT 3a0c3c5; see THIRD_PARTY_NOTICES.md.

"""Training replacements, preserving the verified loss/optimizer arithmetic."""

import torch


def setup_optimizer_and_scheduler(model, cfg):
    from transformers import get_scheduler
    from DiT4DiT.training.train import build_param_lr_groups, dist, logger
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler

def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            # NOTE: In some DeepSpeed/Accelerate configurations, autograd can be globally disabled in engine.forward().
            # We force-enable grad here to ensure `loss.grad_fn` exists in training.

            with torch.enable_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)

                    action_loss = output_dict.get("action_loss", None)
                    future_video_loss = output_dict.get("future_video_loss", None)

                    # Validate at least one loss exists
                    if action_loss is None and future_video_loss is None:
                        raise KeyError(
                            "Model output must contain either `action_loss` or `future_video_loss` for training."
                        )
                    if action_loss is not None and not torch.is_tensor(action_loss):
                        raise TypeError(
                            f"`action_loss` must be a torch.Tensor, got {type(action_loss)}. "
                            "This usually means model.forward returned a Python float or did `.item()`."
                        )
                    if future_video_loss is not None and not torch.is_tensor(future_video_loss):
                        raise TypeError(
                            f"`future_video_loss` must be a torch.Tensor, got {type(future_video_loss)}."
                        )

                    # Compute total loss based on training mode:
                    #   video:  total = future_video_loss * scale
                    #   action: total = action_loss
                    #   joint:  total = action_loss + future_video_loss * scale
                    future_video_loss_scaled = None
                    video_loss_scale = 1.0
                    if future_video_loss is not None:
                        try:
                            video_loss_scale = float(getattr(getattr(self.config.trainer, "loss_scale", None), "future_video", 1.0))
                        except Exception:
                            video_loss_scale = 1.0
                        future_video_loss_scaled = future_video_loss * video_loss_scale

                    if action_loss is not None and future_video_loss is not None:
                        # joint: action + video auxiliary
                        total_loss = action_loss + future_video_loss_scaled
                    elif action_loss is not None:
                        # action only
                        total_loss = action_loss
                    else:
                        # video only
                        total_loss = future_video_loss_scaled

                    # VLA backward propagation (keep inside enable_grad scope)
                    self.accelerator.backward(total_loss)

            # Only step optimizer / scheduler when gradients are synchronized (i.e., end of accumulation window).
            if self.accelerator.sync_gradients:
                # gradient clipping
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

        # Intermediate gradient-accumulation micro-steps are neither logged nor
        # consumed by the outer training loop. Calling .item() here forces the
        # host to wait for the GPU several times per micro-step, so only
        # materialize Python metrics on the real optimizer step. This is after
        # backward/step and does not alter the loss tensor used for training.
        if not self.accelerator.sync_gradients:
            return {}

        step_metrics = {}
        if action_loss is not None:
            step_metrics["action_dit_loss"] = action_loss.item()
        if future_video_loss is not None:
            # raw auxiliary loss
            step_metrics["future_video_loss"] = future_video_loss.item() if torch.is_tensor(future_video_loss) else float(future_video_loss)
        if future_video_loss_scaled is not None:
            step_metrics["future_video_loss_scaled"] = (
                future_video_loss_scaled.item()
                if torch.is_tensor(future_video_loss_scaled)
                else float(future_video_loss_scaled)
            )
        # total loss (for quick monitoring only)
        step_metrics["total_loss"] = total_loss.item() if torch.is_tensor(total_loss) else float(total_loss)
        return step_metrics
