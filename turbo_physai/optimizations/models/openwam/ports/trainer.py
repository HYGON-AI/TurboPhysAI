# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import contextlib
import gzip
import json
import time
import types
import torch
import torch.nn as nn
import openwam.train.openwam_trainer as _source
from turbo_physai.optimizations.common.python.source_binding import bind_missing_globals
from turbo_physai.optimizations.models.openwam import (
    prepare_model,
    prepare_runtime_constants,
)
from turbo_physai.optimizations.models.openwam.settings import value


def OpenWAMTrainer___init__(self, cfg: DictConfig, accelerator=None, dataset=None):
    self.cfg = cfg
    self.dataset = dataset
    self.accelerator = accelerator

    # ---- Reproducible seed (FastWAM-style, yaml-driven) ----
    # Seed before build_architecture so DiT/ActionDiT weight init is
    # deterministic. seed_process uses the same RANK_OFFSET rank stride as
    # per_step_seed and the launcher's seed_everything, so a process's init
    # and per-step seeds share one rank window (cudnn left to the launcher).
    project_cfg = getattr(cfg, "project", None)
    yaml_seed = getattr(project_cfg, "seed", None) if project_cfg is not None else None
    self._rank = int(os.environ.get("RANK", 0))
    self._run_seed = int(yaml_seed) if yaml_seed is not None else None
    if self._run_seed is not None:
        seed_process(self._run_seed, rank=self._rank)
        if self._rank == 0:
            logger.info(
                "Reproducible mode: cfg.project.seed=%d (rank=%d)",
                self._run_seed,
                self._rank,
            )

    t = cfg.training
    m = cfg.model

    # Build architecture (creates video_backbone internally from config).
    from openwam.model import build_architecture, resolve_architecture_config

    # Finetune/resume: build from the self-contained checkpoint dir
    # (skeletons from its config.yaml component specs, weights from its
    # safetensors) so model.video_backbone.model_path need not exist on
    # this host. Weights land here, BEFORE the freeze below. The ckpt config
    # is the reconstruction BASE only — on finetune this run's cfg.model is
    # layered on top (see ckpt_model_loader.merge_ckpt_model_cfg).
    finetune_path = cfg_get(t, "finetune_ckpt_path", None) or None
    resume_path = cfg_get(t, "resume_ckpt_path", None) or None
    if finetune_path and resume_path:
        raise ValueError(
            "finetune_ckpt_path and resume_ckpt_path are mutually exclusive; set at most one."
        )
    self._ckpt_source_dir = finetune_path or resume_path
    if self._ckpt_source_dir is not None:
        from openwam.train.utils.ckpt_model_loader import (
            build_architecture_from_ckpt_dir,
            propagate_component_specs,
            warn_live_model_cfg_ignored,
        )

        # Finetune starts a NEW run: the ckpt config is only the
        # reconstruction base (it alone carries the component/tokenizer
        # specs), and this run's cfg.model overrides it — merged in place, so
        # the architecture built here and the config save_config() writes to
        # the new run dir are the same one. Resume continues ONE run whose
        # config.yaml is already on disk and is reused untouched, so there the
        # ckpt config stays authoritative and divergence is only reported.
        resolved_arch, self.architecture, ckpt_cfg = build_architecture_from_ckpt_dir(
            self._ckpt_source_dir,
            weights_required=finetune_path is not None,
            override_cfg=cfg if finetune_path is not None else None,
        )
        if finetune_path is not None:
            # The merge replaced the cfg.model node, so `m` (bound above)
            # still points at the pre-merge one — rebind it before `freeze`
            # and the architecture logging below read from it.
            m = cfg.model
        else:
            warn_live_model_cfg_ignored(ckpt_cfg, cfg, self._ckpt_source_dir)
            propagate_component_specs(ckpt_cfg, cfg)
    else:
        resolved_arch = resolve_architecture_config(m)
        self.architecture = build_architecture(
            resolved_arch.registry_name, resolved_arch.params
        )
    logger.info(
        "Architecture: %s (framework=%s variant=%s)",
        resolved_arch.registry_name,
        resolved_arch.canonical.framework,
        resolved_arch.canonical.variant,
    )

    # Device placement: skip .to(device) when initialize_model_on_cpu and a real
    # training Accelerator is present — DeepSpeed's prepare() then handles the move.
    _init_on_cpu = bool(t.get("initialize_model_on_cpu", False))
    if not (_init_on_cpu and self.accelerator is not None):
        self.architecture.set_dtype_device(
            self.architecture.dtype, self.architecture.device
        )

    # --- Freeze: declared per-architecture in the model yaml (freeze:);
    # freeze_modules silently skips paths absent on a given architecture.
    freeze_list = list(getattr(m, "freeze", []))
    for name in self.architecture.freeze_modules(freeze_list):
        logger.info("Frozen: %s", name)
    prepare_model(self.architecture)

    # Initialize all schedulers (video + action) inside architecture
    self.architecture.init_training_schedulers(1000)

    # Loss weights from the training config
    self.lambda_video = float(t.lambda_video)
    self.lambda_action = float(t.lambda_action)

    # Push forward-time training flags onto the architecture so prepare_inputs
    # is self-contained.
    self.architecture.set_training_runtime(
        use_gradient_checkpointing=bool(t.use_gradient_checkpointing),
        use_gradient_checkpointing_offload=bool(t.use_gradient_checkpointing_offload),
        max_timestep_boundary=float(t.max_timestep_boundary),
        min_timestep_boundary=float(t.min_timestep_boundary),
    )

    self.model = self  # self-reference some external callers expect

    is_main = self.accelerator is None or self.accelerator.is_main_process
    log_parameter_counts(self.architecture, is_main=is_main)


def OpenWAMTrainer__build_profiler(self, output_path: str):
    """Construct torch.profiler.profile context manager matching DiffSynth-Studio design.

    Controlled by openwam.training.options.profiling (off/rank0/all).

    profile_light=1 disables stacks/shapes and warms up for one
    step (wait=14) while retaining the same active ProfilerStep#15.
    Saves compressed trace directly to:
      <output_path>/profiler_traces/rank{rank}_trace.json.gz
    """
    raw_val = value("profiling", "off").strip().lower()
    if raw_val not in ("rank0", "all"):
        return contextlib.nullcontext()

    if raw_val != "all" and self._rank != 0:
        return contextlib.nullcontext()

    trace_dir = os.path.join(output_path, "profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _sanitize_trace(trace_path: str):
        """Ensure GPU streams and CPU threads remain strictly segregated (泾渭分明).

        Fixes ROCm/HCU Kineto quirk where unmapped background threads (DataLoader
        queues/Inductor pool) during initialization fall back to (pid 0, tid 0),
        which would otherwise rename GPU Stream 0 to 'thread 0 (python3)' and merge
        Python call stacks into the GPU stream track.
        """
        try:
            open_fn = gzip.open if trace_path.endswith(".gz") else open
            with open_fn(trace_path, "rt", encoding="utf-8") as f:
                data = json.load(f)

            events = data.get("traceEvents", [])
            cpu_pid = None
            for e in events:
                if (
                    e.get("name") == "process_labels"
                    and e.get("args", {}).get("labels") == "CPU"
                ):
                    cpu_pid = e.get("pid")
                    break

            modified = False
            for e in events:
                if e.get("pid") == 0 and e.get("cat") == "python_function":
                    if cpu_pid is not None:
                        e["pid"] = cpu_pid
                    modified = True
                elif (
                    e.get("ph") == "M"
                    and e.get("pid") == 0
                    and e.get("tid") == 0
                    and e.get("name") == "thread_name"
                ):
                    if "thread" in e.get("args", {}).get("name", "").lower():
                        e["args"]["name"] = "stream 0 "
                        modified = True

            if modified:
                with open_fn(trace_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f)
        except Exception as err:
            logger.warning("[PROFILER] Trace sanitize warning: %s", err)

    def trace_handler(prof_inst):
        step_tag = (
            f"_step{prof_inst.step_num}" if hasattr(prof_inst, "step_num") else ""
        )
        trace_path = os.path.join(
            trace_dir, f"rank{self._rank}{step_tag}_trace.json.gz"
        )
        t0 = time.monotonic()
        prof_inst.export_chrome_trace(trace_path)
        _sanitize_trace(trace_path)
        duration = time.monotonic() - t0
        msg = f"[PROFILER] Rank {self._rank} exported trace to `{trace_path}` in {duration:.2f}s."
        logger.info(msg)
        if self._rank == 0:
            print(f"\n{msg}\n", flush=True)

    from turbo_physai.optimizations.models.openwam import enabled

    light = enabled("profile_light")
    wait, warmup = (14, 1) if light else (15, 0)
    logger.info(
        "[PROFILER] rank=%d wait=%d warmup=%d active=1 with_stack=%s record_shapes=%s traces=`%s`",
        self._rank,
        wait,
        warmup,
        not light,
        not light,
        trace_dir,
    )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=1, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=not light,
        profile_memory=False,
        with_stack=not light,
        with_modules=False,
    )


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
        logger.info("DEBUG mode: max_steps=20, save@10, constant LR")

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
            logger.info(
                "Skipping DataLoader sampler seed wiring for MixtureDataset; "
                "its index-map shuffle is seeded by the dataset itself."
            )
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
            logger.info(
                "[resume] global_step=%d already complete; finishing.", global_step
            )
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
    prof_context = functools.partial(OpenWAMTrainer__build_profiler, self)(output_path)
    from turbo_physai.optimizations.models.openwam import enabled

    deferred = None
    if enabled("deferred_gather"):
        from turbo_physai.optimizations.models.openwam.deferred_gather import (
            DeferredGather,
        )

        # With ONLINE_ENCODE the encoder's gather replacement launches it.
        deferred = DeferredGather(
            self.accelerator.unwrap_model(self.architecture),
            self.architecture.optimizer,
            patch_step=not enabled("online_encode"),
        )
    online = None
    if enabled("online_encode"):
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
    gc_trace = None
    if enabled("trace_gc"):
        from turbo_physai.optimizations.models.openwam.gc_trace import GCTrace

        gc_trace = GCTrace(output_path, self._rank)
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
        prof_context as prof,
        deferred if deferred is not None else contextlib.nullcontext(),
        online if online is not None else contextlib.nullcontext(),
        gc_trace if gc_trace is not None else contextlib.nullcontext(),
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
                if gc_trace is not None:
                    gc_trace.step(global_step)
                if prof is not None:
                    prof.step()

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


def OpenWAMTrainer_build_optimizer(self) -> torch.optim.Optimizer:
    """AdamW over the per-module (action/video) LR parameter groups."""
    t = self.cfg.training
    params = build_trainable_parameters(
        self,
        action_lr=float(t.action_lr) if getattr(t, "action_lr", None) else None,
        video_lr=float(t.video_lr) if getattr(t, "video_lr", None) else None,
    )
    betas = tuple(getattr(t, "adam_betas", [0.9, 0.95]))
    from turbo_physai.optimizations.models.openwam import enabled

    # One fused kernel per tensor group instead of seven foreach passes over
    # the fp32 ZeRO partition; same AdamW update, rounding may differ slightly.
    extra = {"fused": True} if enabled("fused_adamw") else {}
    return torch.optim.AdamW(
        params,
        lr=float(t.learning_rate),
        weight_decay=float(t.weight_decay),
        betas=betas,
        **extra,
    )


def OpenWAMTrainer_build_dataloader(
    self, batch_size: int
) -> torch.utils.data.DataLoader:
    """Build the training DataLoader.

    ``MixtureDataset`` already owns a shuffled virtual index map and
    reshuffles it in ``set_epoch``.  Do not wrap it in PyTorch's
    ``RandomSampler``: on a large mixture that would build a second full-size
    permutation, convert it to Python integers, and duplicate it in every
    rank. Other datasets keep the standard DataLoader shuffle.

    When ``cfg.project.seed`` is set, wires a per-rank ``generator`` and a
    ``worker_init_fn`` so dataset-side randomness is reproducible across
    runs while keeping per-epoch / per-worker variation.
    """
    from openwam.dataloader.mixture import MixtureDataset

    t = self.cfg.training
    shuffle = not isinstance(self.dataset, MixtureDataset)
    kwargs: dict = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(t.dataset_num_workers),
        collate_fn=list,
        pin_memory=True,
    )
    from turbo_physai.optimizations.models.openwam import enabled

    if enabled("padding_plan_cpu"):
        from turbo_physai.optimizations.models.openwam.text import collate_cpu_padding

        kwargs["collate_fn"] = collate_cpu_padding
    if not shuffle:
        logger.info(
            "DataLoader shuffle disabled for MixtureDataset; its shuffled index map "
            "is reshuffled by dataset.set_epoch(epoch)."
        )
    if self._run_seed is not None:
        from openwam.train.utils.seeding import (
            dataloader_worker_init_fn,
            make_dataloader_generator,
        )

        kwargs["generator"] = make_dataloader_generator(self._run_seed, rank=self._rank)
        kwargs["worker_init_fn"] = dataloader_worker_init_fn
    return torch.utils.data.DataLoader(self.dataset, **kwargs)


def OpenWAMTrainer_prepare_accelerate(self, optimizer, dataloader, scheduler):
    """Wrap architecture/optimizer/dataloader with the DeepSpeed Accelerator."""
    prepare_args = [self.architecture, optimizer, dataloader]
    if scheduler is not None:
        prepare_args.append(scheduler)
        self.architecture, optimizer, dataloader, scheduler = self.accelerator.prepare(
            *prepare_args
        )
    else:
        self.architecture, optimizer, dataloader = self.accelerator.prepare(
            *prepare_args
        )
    # After prepare, self.architecture is the wrapped handle used for the loop's
    # forward/backward. Architecture-level helpers must run on the UNDERLYING
    # module, not the wrapper: DeepSpeedEngine.__getattr__ forwards unknown attrs
    # to the inner module, but DistributedDataParallel does NOT — so calling
    # set_dtype_device/move_frozen_to_device on the wrapper would raise
    # AttributeError under a hypothetical plain-DDP accelerator (world_size>1).
    # unwrap_model returns the inner module for both backends (no-op single-GPU,
    # where Accelerate adds no wrapper). NB: this only covers these setup-time
    # helpers — the training loop's self.architecture.prepare_inputs/compute_loss
    # calls would hit the same non-forwarding wrapper if a non-DeepSpeed multi-GPU
    # path is ever reintroduced.
    arch = self.accelerator.unwrap_model(self.architecture)
    # Propagate device down through architecture; frozen modules (T5/VAE) idempotent move.
    arch.set_dtype_device(arch.dtype, self.accelerator.device)
    from turbo_physai.optimizations.models.openwam import enabled

    if enabled("vae_latent_cache"):
        # Stored latents replace the VAE. DeepSpeed moved the whole module to
        # the device; return the frozen VAE to the CPU and release its memory.
        arch.move_frozen_to_device(self.accelerator.device, names=("text_encoder",))
        arch.video_backbone.vae.to(device="cpu")
        torch.cuda.empty_cache()
        logger.info("[optimizations] VAE kept on CPU; training reads stored latents")
    else:
        arch.move_frozen_to_device(self.accelerator.device)
    from turbo_physai.optimizations.models.openwam import text_store

    if text_store.requested():
        text_store.offload(arch.video_backbone)
        torch.cuda.empty_cache()
        logger.info("[optimizations] T5 weights moved to CPU for offline text caching")
    prepare_runtime_constants(arch)

    if enabled("zero_reduce_scatter"):
        from turbo_physai.optimizations.models.openwam.zero import (
            install_reduce_scatter,
        )

        install_reduce_scatter(self.architecture.optimizer)
    logger.info(
        "architecture wrapped (%s), device=%s",
        type(self.architecture).__name__,
        self.accelerator.device,
    )
    return optimizer, dataloader, scheduler


def OpenWAMTrainer_compute_loss(self, batch) -> dict:
    """Compute joint video-action loss. Returns dict: total/video/action."""
    from turbo_physai.optimizations.models.openwam import enabled

    if enabled("checkpoint_stats"):
        from turbo_physai.optimizations.models.openwam.sac import begin_step

        begin_step(self.accelerator.device)
    from turbo_physai.optimizations.models.openwam import deferred_gather
    from turbo_physai.optimizations.models.openwam.online_encode import PreparedInputs

    # deferred_gather: non-layer parameters may still be in flight.
    deferred_gather.wait_prologue()
    if isinstance(batch, PreparedInputs):
        inputs = batch.values
    else:
        if not isinstance(batch, list):
            batch = [batch]
        inputs = self.architecture.prepare_inputs(batch)
    if self.lambda_action > 0 and inputs.get("actions") is None:
        raise ValueError("lambda_action > 0 but no action in data.")

    result = self.architecture.compute_loss(
        **inputs,
        lambda_video=self.lambda_video,
        lambda_action=self.lambda_action,
    )

    return {
        "total": result["loss"],
        "video": result.get("loss_video", torch.tensor(0.0)),
        "action": result.get("loss_action", torch.tensor(0.0)),
    }


def OpenWAMTrainer_log_step(
    self,
    *,
    metrics,
    global_step,
    opt_step,
    epoch,
    lr,
    steps_per_sec,
    batch_size,
    total_steps,
    wandb_run,
    debug,
    output_path,
) -> None:
    """Print one rank-0 line per reported step, log to WandB, and retain debug CSV output."""
    from turbo_physai.optimizations.models.openwam import enabled

    if enabled("checkpoint_stats"):
        from turbo_physai.optimizations.models.openwam.sac import report_step

        report_step(
            global_step,
            self.accelerator.device,
            metrics,
            steps_per_sec,
            batch_size,
            opt_step,
        )
    # Metrics are already reduced by every rank before entering this method.
    # Gate only output; no collective or training work belongs behind this check.
    is_main = self.accelerator is None or self.accelerator.is_main_process
    if not is_main:
        return
    labels = [("action", "loss_action")]
    loss_total = metrics["loss_total"]
    loss_video = metrics["loss_video"]
    grad_norm = metrics["grad_norm"]

    num_procs = self.accelerator.num_processes if self.accelerator is not None else 1
    samples_per_sec = steps_per_sec * batch_size * num_procs
    loss_parts = " ".join(f"{name}={metrics[key]:.6f}" for name, key in labels)
    tag = "debug" if debug else "train"
    msg = (
        f"[{tag}][step {global_step:04d}/{total_steps:04d} opt {opt_step:04d}] "
        f"loss={loss_total:.6f} video={loss_video:.6f} {loss_parts} "
        f"grad_norm={grad_norm:.6f} lr={lr:.3e} epoch={epoch} "
        f"s/step={1.0 / max(steps_per_sec, 1e-9):.4f} samples/s={samples_per_sec:.3f}"
    )
    # Like parameter counts, print directly so Hydra's logging filter cannot
    # hide progress. Plain newlines also remain readable through tee.
    print(msg, flush=True)

    if wandb_run is not None:
        log_dict = {
            "train/loss": loss_total,
            "train/loss_video": loss_video,
            "train/grad_norm": grad_norm,
            "train/lr": lr,
            "performance/steps_per_sec": steps_per_sec,
            "performance/samples_per_sec": samples_per_sec,
        }
        for name, key in labels:
            log_dict[f"train/loss_{name}"] = metrics[key]
        wandb_run.log(log_dict, step=global_step)

    if debug and output_path:
        write_debug_loss_row(
            output_path,
            labels=labels,
            metrics=metrics,
            global_step=global_step,
            opt_step=opt_step,
            epoch=epoch,
            lr=lr,
            steps_per_sec=steps_per_sec,
        )


def OpenWAMTrainer_finish_training(
    self,
    output_path: str,
    global_step: int,
    save_steps,
    keep_last_k: int,
    is_main: bool,
    wandb_run,
) -> None:
    """Unified teardown for every exit path: final weights, drop resume state, close wandb.

    Falsy ``save_steps`` = a profiling/no-write run, so no final artifact (matches the
    periodic-save gating). After the final weights land, ``finalize_keep_weights_only``
    removes every ``accel_state_step_*`` and retains the configured number of recent
    weight checkpoints (rank-0, post-barrier).
    """
    from turbo_physai.optimizations.models.openwam import deferred_gather

    deferred_gather.finish()
    if save_steps:
        save_weights(
            self.accelerator, self.architecture, output_path, global_step, final=True
        )
    self.accelerator.wait_for_everyone()
    if is_main:
        finalize_keep_weights_only(output_path, keep_last_k)
    if wandb_run is not None:
        wandb_run.finish()


bind_missing_globals(globals(), _source)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer___init___wrapper = _port_wrapper(
    OpenWAMTrainer___init__,
    "openwam.train.openwam_trainer.OpenWAMTrainer.__init__",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_train_wrapper = _port_wrapper(
    OpenWAMTrainer_train,
    "openwam.train.openwam_trainer.OpenWAMTrainer.train",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_build_optimizer_wrapper = _port_wrapper(
    OpenWAMTrainer_build_optimizer,
    "openwam.train.openwam_trainer.OpenWAMTrainer.build_optimizer",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_build_dataloader_wrapper = _port_wrapper(
    OpenWAMTrainer_build_dataloader,
    "openwam.train.openwam_trainer.OpenWAMTrainer.build_dataloader",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_prepare_accelerate_wrapper = _port_wrapper(
    OpenWAMTrainer_prepare_accelerate,
    "openwam.train.openwam_trainer.OpenWAMTrainer.prepare_accelerate",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_compute_loss_wrapper = _port_wrapper(
    OpenWAMTrainer_compute_loss,
    "openwam.train.openwam_trainer.OpenWAMTrainer.compute_loss",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_log_step_wrapper = _port_wrapper(
    OpenWAMTrainer_log_step,
    "openwam.train.openwam_trainer.OpenWAMTrainer.log_step",
    "openwam.training",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

OpenWAMTrainer_finish_training_wrapper = _port_wrapper(
    OpenWAMTrainer_finish_training,
    "openwam.train.openwam_trainer.OpenWAMTrainer.finish_training",
    "openwam.training",
)
