# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

"""Performance scheduling only; setup, logs and checkpoint teardown stay native."""

from __future__ import annotations

import contextlib
import functools
import gzip
import json
import os
import subprocess
import time as _time
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
from openwam.train.utils.grad_accumulation import scale_loss_for_backward
from openwam.train.utils.runtime_tuning import maybe_freeze_gc
from tqdm import tqdm

from .. import prepare_model, prepare_runtime_constants
from ..settings import enabled, port_wrapper


def _build_profiler(trainer, output_path: str):
    """Construct torch.profiler.profile context manager matching DiffSynth-Studio design.

    Controlled cleanly by single environment variable:
      - enable_profiling=true (or 1): profiles rank 0 (wait=15, warmup=0, active=2, repeat=1).
      - enable_profiling=all:        profiles all ranks.

    OPENWAM_PROFILE_LIGHT=1 disables stacks/shapes and warms up for one
    step (wait=14) while retaining the same active ProfilerStep#15.
    Saves compressed trace directly to:
      <output_path>/profiler_traces/rank{rank}_trace.json.gz
    """
    raw_val = (
        os.environ.get("enable_profiling") or os.environ.get("ENABLE_PROFILING", "")
    ).strip().lower()
    if raw_val not in ("true", "1", "all", "yes", "on"):
        return contextlib.nullcontext()

    rank = getattr(trainer, "_rank", 0)
    if raw_val != "all" and rank != 0:
        return contextlib.nullcontext()

    trace_dir = os.path.join(output_path, "profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def _sanitize_trace(trace_path: str):
        """Ensure GPU streams and CPU threads remain strictly segregated (泾渭分明).

        Fixes ROCm/DCU Kineto quirk where unmapped background threads (DataLoader
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
                if e.get("name") == "process_labels" and e.get("args", {}).get("labels") == "CPU":
                    cpu_pid = e.get("pid")
                    break

            modified = False
            for e in events:
                if e.get("pid") == 0 and e.get("cat") == "python_function":
                    if cpu_pid is not None:
                        e["pid"] = cpu_pid
                    modified = True
                elif e.get("ph") == "M" and e.get("pid") == 0 and e.get("tid") == 0 and e.get("name") == "thread_name":
                    if "thread" in e.get("args", {}).get("name", "").lower():
                        e["args"]["name"] = "stream 0 "
                        modified = True

            if modified:
                with open_fn(trace_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f)
        except Exception as err:
            logger.warning("[PROFILER] Trace sanitize warning: %s", err)

    def trace_handler(prof_inst):
        step_tag = f"_step{prof_inst.step_num}" if hasattr(prof_inst, "step_num") else ""
        trace_path = os.path.join(trace_dir, f"rank{rank}{step_tag}_trace.json.gz")
        t0 = _time.monotonic()
        prof_inst.export_chrome_trace(trace_path)
        _sanitize_trace(trace_path)
        duration = _time.monotonic() - t0
        msg = f"[PROFILER] Rank {rank} exported trace to `{trace_path}` in {duration:.2f}s."
        logger.info(msg)
        if rank == 0:
            print(f"\n{msg}\n", flush=True)

    light = os.environ.get("OPENWAM_PROFILE_LIGHT", "").strip().lower() in ("1", "true", "yes", "on")
    wait, warmup = (14, 1) if light else (15, 0)
    logger.info(
        "[PROFILER] rank=%d wait=%d warmup=%d active=2 with_stack=%s record_shapes=%s traces=`%s`",
        rank,
        wait,
        warmup,
        not light,
        not light,
        trace_dir,
    )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=2, repeat=1),
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
            pbar=pbar,
            wandb_run=wandb_run,
            debug=debug,
            output_path=output_path,
        )

    def flush_metrics():
        if lagged is not None:
            for info, metrics in lagged.flush():
                report(info, metrics)

    prof_context = (
        self._build_profiler(output_path)
        if hasattr(self, "_build_profiler")
        else _build_profiler(self, output_path)
    )

    with (
        prof_context as prof,
        deferred if deferred is not None else contextlib.nullcontext(),
        online if online is not None else contextlib.nullcontext(),
        tqdm(
            total=total_steps,
            desc="Training",
            unit="step",
            initial=min(global_step, total_steps),
        ) as pbar,
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
                # Accelerate's rebuilt loader may drop the asynchronous transfer setting.
                if getattr(dataloader, "_non_blocking", False) and hasattr(epoch_iter, "_non_blocking"):
                    epoch_iter._non_blocking = True
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
                    self.accelerator.backward(
                        scale_loss_for_backward(self.accelerator, loss)
                    )

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
                maybe_freeze_gc(global_step)  # OPENWAM_GC_FREEZE
                if prof is not None and hasattr(prof, "step"):
                    prof.step()

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


def _gpu_numa_cpus():
    """Hygon / ROCm NUMA node and CPU detection with nvidia-smi fallback."""
    from openwam.train.utils.runtime_tuning import _parse_cpulist

    local = int(os.environ.get("LOCAL_RANK", "0"))
    visible = (
        os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("ROCR_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    phys = visible.split(",")[local].strip() if visible else str(local)

    node = None
    drm_path = f"/sys/class/drm/card{phys}/device/numa_node"
    if os.path.exists(drm_path):
        try:
            with open(drm_path) as f:
                val = int(f.read().strip())
                if val >= 0:
                    node = val
        except Exception:
            pass

    if node is None:
        try:
            out = subprocess.run(
                ["rocm-smi", "-d", str(phys), "--showbus", "--csv"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            for line in out.splitlines():
                if "PCI Bus" in line or "," in line:
                    parts = [p.strip() for p in line.split(",")]
                    for part in parts:
                        if part.count(":") >= 1:
                            bus_str = part.lower()
                            if bus_str.startswith("0x"):
                                bus_str = bus_str[2:]
                            pci_path = f"/sys/bus/pci/devices/{bus_str}/numa_node"
                            if os.path.exists(pci_path):
                                with open(pci_path) as f:
                                    val = int(f.read().strip())
                                    if val >= 0:
                                        node = val
                                        break
        except Exception:
            pass

    if node is None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            bus = None
            for line in out.splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 3:
                    idx, uuid, pci = parts[0], parts[1], parts[2]
                    if phys in (idx, uuid):
                        bus = pci
            if bus is not None:
                dom, rest = bus.split(":", 1)
                sysname = f"{dom[-4:]}:{rest}".lower()
                numa_file = f"/sys/bus/pci/devices/{sysname}/numa_node"
                if os.path.exists(numa_file):
                    with open(numa_file) as f:
                        val = int(f.read().strip())
                        if val >= 0:
                            node = val
        except Exception:
            pass

    if node is None or node < 0:
        return None
    cpulist_path = f"/sys/devices/system/node/node{node}/cpulist"
    if not os.path.exists(cpulist_path):
        return None
    with open(cpulist_path) as f:
        return node, _parse_cpulist(f.read())


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
            defaults = {**optimizer.defaults, "fused": True, "foreach": False}
            # AdamW injects this into Adam's defaults on newer PyTorch, but
            # does not accept it in its own constructor.
            defaults.pop("decoupled_weight_decay", None)
            optimizer = torch.optim.AdamW(
                [
                    dict(group, fused=True, foreach=False)
                    for group in optimizer.param_groups
                ],
                **defaults,
            )
        return optimizer

    return wrapped


def build_dataloader(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        loader = original(self, *args, **kwargs)
        if enabled("padding_plan_cpu"):
            from openwam.dataloader.collate import openwam_collate
            from ..text import collate_cpu_padding, collate_cpu_padding_uint8

            if loader.collate_fn is openwam_collate:
                loader.collate_fn = collate_cpu_padding_uint8
            elif loader.collate_fn is None or loader.collate_fn is list:
                loader.collate_fn = collate_cpu_padding
        return loader

    return wrapped


def prepare_accelerate(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if enabled("host_async"):
            self.accelerator.dataloader_config.non_blocking = True
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

_gpu_numa_cpus_wrapper = port_wrapper(
    _gpu_numa_cpus,
    "openwam.train.utils.runtime_tuning._gpu_numa_cpus",
    "openwam.training",
)
