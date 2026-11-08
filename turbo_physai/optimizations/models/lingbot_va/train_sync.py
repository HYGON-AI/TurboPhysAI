# Adapted from Robbyant/lingbot-va 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb, Apache-2.0.
import functools
import gc
import torch
import torch.distributed as dist
from tqdm import tqdm
from train import logger, dist_mean, dist_max

def train(self):
    """Main training loop - train by steps instead of epochs."""
    logger.info(f"Starting training for {self.config.num_steps} steps...")
    self.transformer.train()

    progress_bar = tqdm(
        total=self.config.num_steps,
        desc="Training",
        disable=(self.config.rank != 0),
        leave=True,
        dynamic_ncols=True,
        initial=self.step
    )

    self.optimizer.zero_grad()
    accumulated_latent_losses = []
    accumulated_action_losses = []
    step_in_accumulation = 0

    while self.step < self.config.num_steps:
        # Get next batch (handles epoch reset automatically)
        batch = self._get_next_batch()

        losses = self._train_step(batch, step_in_accumulation)

        # Accumulate losses for logging
        accumulated_latent_losses.append(losses['latent_loss'])
        accumulated_action_losses.append(losses['action_loss'])
        step_in_accumulation += 1

        # Log and checkpoint when optimizer steps
        if losses['should_log']:
            lr = self.lr_scheduler.get_last_lr()[0]

            # Average accumulated losses
            latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
            action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
            max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
            max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

            # Clear accumulated losses
            accumulated_latent_losses = []
            accumulated_action_losses = []
            step_in_accumulation = 0

            torch.cuda.synchronize()
            if self.step % self.config.gc_interval == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if self.config.rank == 0:
                total_norm = losses['total_norm']
                progress_bar.n += 1
                progress_bar.set_postfix({
                    'latent_loss': f'{latent_loss_show:.4f}',
                    'action_loss': f'{action_loss_show:.4f}',
                    'step': self.step,
                    'grad_norm': f'{total_norm.item():.2f}',
                    'lr': f'{lr:.2e}'
                })
                if self.config.enable_wandb:
                    self.wandb.log({
                        'loss_metrics/global_avg_video_loss': latent_loss_show,
                        'loss_metrics/global_avg_action_loss': action_loss_show,
                        'loss_metrics/global_max_video_loss': max_latent_loss_show,
                        'loss_metrics/global_max_action_loss': max_action_loss_show,
                        'grad_norm': total_norm.item(),
                        'lr': lr,
                    }, step=self.step)

            self.step += 1

            if self.step % self.config.save_interval == 0:
                if self.config.rank == 0:
                    logger.info(f"Starting save model at step {self.step}")
                self.save_checkpoint()

        if losses['should_log'] and dist.is_initialized():
            dist.barrier()

    progress_bar.close()
    logger.info("Training completed!")

def sync_wrapper(original, options):
    if options:raise ValueError('train_sync accepts no options')
    @functools.wraps(original)
    def wrapped(self):
        if self.gradient_accumulation_steps<=1:return original(self)
        return train(self)
    return wrapped
