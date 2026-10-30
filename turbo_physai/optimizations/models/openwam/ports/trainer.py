# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

"""Performance scheduling only; setup, logs and checkpoint teardown stay native."""

from __future__ import annotations

import contextlib
import functools
import torch
from openwam.train.openwam_trainer import (
    cfg_get,
    init_wandb,
    itertools,
    logger,
    manage_checkpoints,
    math,
    per_step_seed,
    reduce_step_metrics,
    save_full_state,
    save_weights,
    wire_sampler_seed,
)
from .. import prepare_model, prepare_runtime_constants
from ..settings import enabled, port_wrapper


def OpenWAMTrainer_train(self, num_epochs: int = None, max_steps: int = None):
    """Run the training loop (HuggingFace Accelerate distributed).

    Three entry modes (train.yaml finetune/resume fields): fresh, finetune warm-start
    (weights loaded at construction), or resume (load full state after prepare).
    """
    t = self.cfg.training
    num_epochs = num_epochs or cfg_get(t, "num_epochs", None)
    num_epochs = int(num_epochs) if num_epochs is not None else None
    max_steps = max_steps or getattr(t, "max_steps", None)
    batch_size = int(t.batch_size)
    grad_accum = int(t.gradient_accumulation_steps)

    debug = bool(getattr(t, "debug", False))
    if debug:
        max_steps = 20
        save_steps_override = 10

    # num_epochs=null means step-only training: max_steps is the sole stop condition.
    if num_epochs is None and not max_steps:
        raise ValueError(
            "training.num_epochs and training.max_steps are both unset; "
            "set num_epochs, or set max_steps for step-only training."
        )

    # finetune/resume validation + architecture construction happen in
    # __init__ (self-contained ckpt-dir path); only resume needs a path here.
    resume_path = cfg_get(t, "resume_ckpt_path", None) or None

    optimizer = self.build_optimizer()
    dataloader = self.build_dataloader(batch_size)
    max_grad_norm = (
        float(t.max_grad_norm) if getattr(t, "max_grad_norm", None) else None
    )

    n_proc = self.accelerator.num_processes
    steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
    # max_steps counts micro-steps (global_step); convert to optimizer steps for the LR horizon.
    max_opt_steps = math.ceil(max_steps / grad_accum) if max_steps else None
    if num_epochs is not None:
        # prepare() shards the dataloader ~1/n_proc; fold that in so total_opt_steps is the
        # per-process optimizer steps the loop actually runs (same unit as max_opt_steps).
        steps_per_epoch = math.ceil(steps_per_epoch / n_proc)
        total_opt_steps = steps_per_epoch * num_epochs
        if max_opt_steps:
            total_opt_steps = min(total_opt_steps, max_opt_steps)
    else:
        total_opt_steps = max_opt_steps
    scheduler = self.build_lr_scheduler(optimizer, total_opt_steps, debug=debug)

    if debug:
        save_steps = save_steps_override
    else:
        save_steps = getattr(t, "save_steps", None)
        if save_steps is not None:
            save_steps = int(save_steps)
    keep_last_k = int(getattr(t, "keep_last_k_ckpts", 3))
    save_full_states_for_resume = bool(getattr(t, "save_full_states_for_resume", False))

    # Finetune warm-start weights were already loaded at architecture
    # construction (__init__, self-contained ckpt-dir path). Step stays 0.

    output_path, resume_state_dir = self.setup_output_dir(debug, resume_path)
    optimizer, dataloader, scheduler = self.prepare_accelerate(
        optimizer, dataloader, scheduler
    )

    if self._run_seed is not None:
        from openwam.dataloader.mixture import MixtureDataset

        if isinstance(self.dataset, MixtureDataset):
            pass
        else:
            wire_sampler_seed(dataloader, int(self._run_seed), rank=self._rank)

    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    is_main = self.accelerator is None or self.accelerator.is_main_process
    wandb_run = None if (debug or not is_main) else init_wandb(self.cfg)

    if num_epochs is not None:
        total_steps = len(dataloader) * num_epochs
        if max_steps:
            total_steps = min(total_steps, max_steps)
    else:
        total_steps = max_steps

    import time as _time

    opt_step = 0
    global_step = 0
    start_epoch = 0
    skip_first = 0

    # Resume: restore full state AFTER prepare, then map global_step -> (epoch, skip).
    if resume_state_dir is not None:
        global_step, opt_step, start_epoch, skip_first = self.resume_if_configured(
            resume_state_dir, dataloader, grad_accum
        )
        already_done = (num_epochs is not None and start_epoch >= num_epochs) or (
            max_steps and global_step >= max_steps
        )
        if already_done:
            self.finish_training(
                output_path, global_step, save_steps, keep_last_k, is_main, wandb_run
            )
            return
        if skip_first > 0 and self._run_seed is None:
            logger.warning(
                "[resume] mid-epoch resume (skip_first=%d) without project.seed: DataLoader "
                "shuffle is non-reproducible, so the resumed epoch's batch order differs from "
                "the original run — samples may be silently re-fed or skipped. Set project.seed "
                "for faithful mid-epoch resume.",
                skip_first,
            )

    from turbo_physai.optimizations.models.openwam import enabled

    if enabled("vae_latent_cache"):
        from turbo_physai.optimizations.models.openwam.latent_cache import (
            attach,
            vae_fingerprint,
            validate_backbone,
        )

        # After finetune/resume weights are in place and before DataLoader
        # workers fork (they start at the first iteration).
        arch = self.accelerator.unwrap_model(self.architecture)
        validate_backbone(arch)
        attach(self.dataset, vae_fingerprint(arch.video_backbone.vae))

    from turbo_physai.optimizations.models.openwam import text_store

    if text_store.requested():
        # Resume has restored weights by this point. Never serve a cache
        # fingerprinted against the pre-resume encoder.
        arch = self.accelerator.unwrap_model(self.architecture)
        text_store.attach(self.dataset, arch.video_backbone)

    _step_t0 = _time.monotonic()

    assert self.accelerator is not None, "OpenWAMTrainer requires an Accelerator"

    # Per-step manual_seed makes timestep + noise sampling in compute_loss
    # reproducible across runs and ZeRO stages: same (rank, step) -> same RNG,
    # different ranks at the same step keep in-batch timestep diversity.
    # Gated on _run_seed so unseeded production runs stay fully stochastic.
    from turbo_physai.optimizations.models.openwam import enabled

    online_eligible = (
        enabled("online_encode") and grad_accum == 1 and int(dataloader.num_workers) > 0
    )
    deferred = None
    if enabled("deferred_gather"):
        from turbo_physai.optimizations.models.openwam.deferred_gather import (
            DeferredGather,
        )

        # With ONLINE_ENCODE the encoder's gather replacement launches it.
        deferred = DeferredGather(
            self.accelerator.unwrap_model(self.architecture),
            self.architecture.optimizer,
            patch_step=not online_eligible,
        )
    online = None
    if online_eligible:
        from turbo_physai.optimizations.models.openwam.online_encode import (
            OnlineEncoder,
        )

        online = OnlineEncoder(
            self.accelerator.unwrap_model(self.architecture),
            self.architecture.optimizer,
            num_workers=dataloader.num_workers,
            gradient_accumulation_steps=grad_accum,
            deferred=deferred,
        )
    lagged = None
    if enabled("host_async"):
        from turbo_physai.optimizations.models.openwam.host_async import (
            LaggedMetrics,
            device_scalar,
        )

        lagged = LaggedMetrics(self.accelerator)

    def report(info, metrics):
        self.log_step(
            metrics=metrics,
            **info,
            batch_size=batch_size,
            total_steps=total_steps,
            wandb_run=wandb_run,
            debug=debug,
            output_path=output_path,
        )

    def flush_metrics():
        if lagged is not None:
            for info, metrics in lagged.flush():
                report(info, metrics)

    with (
        deferred if deferred is not None else contextlib.nullcontext(),
        online if online is not None else contextlib.nullcontext(),
    ):
        epochs = (
            itertools.count(start_epoch)
            if num_epochs is None
            else range(start_epoch, num_epochs)
        )
        for epoch in epochs:
            if hasattr(dataloader, "set_epoch"):
                dataloader.set_epoch(epoch)
            if hasattr(self.dataset, "set_epoch"):
                self.dataset.set_epoch(epoch)
            # On the resumed epoch, skip the batches already consumed before the checkpoint.
            if epoch == start_epoch and skip_first > 0:
                from accelerate import skip_first_batches

                epoch_iter = skip_first_batches(dataloader, skip_first)
            else:
                epoch_iter = dataloader
            if online is not None:
                epoch_iter = online.batches(
                    epoch_iter,
                    global_step=global_step,
                    max_steps=max_steps,
                    save_steps=save_steps,
                )
            for batch in epoch_iter:
                if self._run_seed is not None:
                    step_seed = per_step_seed(
                        self._run_seed, rank=self._rank, step=global_step
                    )
                    torch.manual_seed(step_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(step_seed)
                with self.accelerator.accumulate(self.architecture):
                    losses = self.compute_loss(batch)
                    loss = losses["total"]
                    self.accelerator.backward(loss)

                    # DeepSpeed may already have queued step-end AllGather
                    # inside backward. A host scalar upload would wait for it.
                    grad_norm = (
                        torch.zeros((), device=loss.device, dtype=torch.float32)
                        if lagged is not None
                        else torch.tensor(0.0, device=loss.device)
                    )
                    if self.accelerator.sync_gradients:
                        if max_grad_norm is not None:
                            from turbo_physai.optimizations.models.openwam.host_async import (
                                device_grad_norm,
                            )

                            grad_norm_val = (
                                device_grad_norm(self.accelerator)
                                if lagged is not None
                                else None
                            )
                            if grad_norm_val is None:
                                grad_norm_val = self.accelerator.clip_grad_norm_(
                                    all_params, max_grad_norm
                                )
                            if lagged is not None:
                                grad_norm = device_scalar(grad_norm_val, loss.device)
                            else:
                                grad_norm = torch.tensor(
                                    float(grad_norm_val), device=loss.device
                                )
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()
                        opt_step += 1

                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                _now = _time.monotonic()
                steps_per_sec = 1.0 / max(_now - _step_t0, 1e-9)
                _step_t0 = _now
                info = dict(
                    global_step=global_step,
                    opt_step=opt_step,
                    epoch=epoch,
                    lr=current_lr,
                    steps_per_sec=steps_per_sec,
                )
                if lagged is None:
                    report(
                        info, reduce_step_metrics(self.accelerator, losses, grad_norm)
                    )
                else:
                    # Logged one step late so the host never waits for this step's values.
                    for ready_info, metrics in lagged.push(losses, grad_norm, info):
                        report(ready_info, metrics)

                # save_steps: write the weights line (+ the resumable full state when
                # save_full_states_for_resume=true), then prune in lockstep.
                if save_steps and global_step > 0 and global_step % save_steps == 0:
                    if deferred is not None:
                        deferred.finish()
                    save_weights(
                        self.accelerator,
                        self.architecture,
                        output_path,
                        global_step,
                        final=False,
                    )
                    if save_full_states_for_resume:
                        save_full_state(
                            self.accelerator, output_path, global_step, opt_step, epoch
                        )
                    if is_main:
                        manage_checkpoints(output_path, keep_last_k)

                if max_steps and global_step >= max_steps:
                    flush_metrics()
                    self.finish_training(
                        output_path,
                        global_step,
                        save_steps,
                        keep_last_k,
                        is_main,
                        wandb_run,
                    )
                    return

        flush_metrics()
        self.finish_training(
            output_path, global_step, save_steps, keep_last_k, is_main, wandb_run
        )


def trainer_init(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        from ..launch import configure_backends

        configure_backends()
        result = original(self, *args, **kwargs)
        prepare_model(self.architecture)
        return result

    wrapped.__openwam_group__ = "openwam.training"
    wrapped.__openwam_options__ = options
    return wrapped


def build_optimizer(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        optimizer = original(self, *args, **kwargs)
        if enabled("fused_adamw"):
            # Native build_optimizer creates torch.optim.AdamW. Keep parameter
            # grouping, master weights and hyperparameters owned by OpenWAM.
            if type(optimizer) is not torch.optim.AdamW:
                raise TypeError("fused_adamw requires native torch.optim.AdamW")
            if optimizer.state:
                raise ValueError(
                    "fused_adamw must be configured before optimizer state is initialized"
                )
            for group in optimizer.param_groups:
                if group.get("differentiable"):
                    raise ValueError(
                        "fused_adamw does not support differentiable AdamW"
                    )
                group.update(fused=True, foreach=False)
            optimizer = torch.optim.AdamW(
                optimizer.param_groups,
                **{**optimizer.defaults, "fused": True, "foreach": False},
            )
        return optimizer

    return wrapped


def build_dataloader(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        loader = original(self, *args, **kwargs)
        if enabled("padding_plan_cpu"):
            from ..text import collate_cpu_padding

            loader.collate_fn = collate_cpu_padding
        return loader

    return wrapped


def prepare_accelerate(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        arch = self.accelerator.unwrap_model(self.architecture)
        if enabled("vae_latent_cache"):
            arch.video_backbone.vae.to(device="cpu")
            torch.cuda.empty_cache()
        from .. import text_store

        if text_store.requested():
            text_store.offload(arch.video_backbone)
            torch.cuda.empty_cache()
        prepare_runtime_constants(arch)
        if enabled("zero_reduce_scatter"):
            from ..zero import install_reduce_scatter

            install_reduce_scatter(self.architecture.optimizer)
        return result

    return wrapped


def compute_loss(original, options):
    @functools.wraps(original)
    def wrapped(self, batch):
        from .. import deferred_gather
        from ..online_encode import PreparedInputs

        deferred_gather.wait_prologue()
        if not isinstance(batch, PreparedInputs):
            return original(self, batch)
        inputs = batch.values
        if self.lambda_action > 0 and inputs.get("actions") is None:
            raise ValueError("lambda_action > 0 but no action in data.")
        result = self.architecture.compute_loss(
            **inputs, lambda_video=self.lambda_video, lambda_action=self.lambda_action
        )
        return {
            "total": result["loss"],
            "video": result.get("loss_video", torch.tensor(0.0)),
            "action": result.get("loss_action", torch.tensor(0.0)),
        }

    return wrapped


def finish_training(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        from .. import deferred_gather

        deferred_gather.finish()
        return original(self, *args, **kwargs)

    return wrapped


OpenWAMTrainer_train_wrapper = port_wrapper(
    OpenWAMTrainer_train,
    "openwam.train.openwam_trainer.OpenWAMTrainer.train",
    "openwam.training",
)
