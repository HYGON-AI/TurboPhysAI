# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""DistributedCheckpointer HIP adaptations (adapted source port).

- ``_get_dcp_metadata_process_group``: lazy CPU/Gloo group for DCP save-plan
  object collectives (RCCL gather_object instability), controlled by
  ``COSMOS_DCP_METADATA_BACKEND`` (auto|gloo|default).
- ``get_storage_writer``: ``COSMOS_DCP_PER_THREAD_COPY_AHEAD_BYTES`` (auto -> 0
  on HIP) selects serial device-to-CPU staging in FileSystemWriter.
- ``_checkpoint_async_with_pinned_memory``: fall back to a synchronous save when
  shared-memory staging allocation fails.
- ``save_state_dict_worker``: thread the metadata group into ``dcp.save``.
- ``load``: RNG-state tolerant restore + net_ema warm-start reseed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple, Union

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

import cosmos_framework.checkpoint.dcp as _dcp_mod
from cosmos_framework.utils import log, misc

from ._source_binding import bind_missing_globals


def checkpointer_post_init(original, options):
    """Wrap ``DistributedCheckpointer.__init__`` to add the metadata-group slot."""
    del options

    def initialized(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self._dcp_metadata_process_group = None
        return result

    return initialized


def _get_dcp_metadata_process_group(self) -> dist.ProcessGroup | None:
    """Return a CPU metadata group for HCU DCP save-plan collectives."""
    if self._dcp_metadata_process_group is not None:
        return self._dcp_metadata_process_group
    if not dist.is_initialized():
        return None

    configured_backend = os.getenv("COSMOS_DCP_METADATA_BACKEND", "auto").strip().lower()
    if configured_backend in {"default", "none", "disabled"}:
        return None

    default_backend = str(dist.get_backend()).lower()
    use_gloo = configured_backend == "gloo" or (
        configured_backend == "auto"
        and torch.version.hip is not None
        and default_backend in {"nccl", "rccl"}
    )
    if not use_gloo:
        return None

    self._dcp_metadata_process_group = dist.new_group(backend="gloo")
    log.info(
        "DCP save-plan metadata collectives use a dedicated Gloo process group "
        f"(default backend={default_backend})"
    )
    return self._dcp_metadata_process_group


def get_storage_writer(self, checkpoint_path: str) -> Union[S3StorageWriter, FileSystemWriter]:
    if self.save_to_object_store:
        return S3StorageWriter(
            credential_path=self.config_checkpoint.save_to_object_store.credentials,
            path=checkpoint_path,
            enable_gcs_patch_in_boto3=self.config_checkpoint.enable_gcs_patch_in_boto3,
        )
    configured_copy_ahead = os.getenv(
        "COSMOS_DCP_PER_THREAD_COPY_AHEAD_BYTES", "auto"
    ).strip().lower()
    if configured_copy_ahead in {"default", "none", "disabled"}:
        copy_ahead_bytes = None
    elif configured_copy_ahead == "auto":
        copy_ahead_bytes = 0 if torch.version.hip is not None else None
    else:
        try:
            copy_ahead_bytes = int(configured_copy_ahead)
        except ValueError as exc:
            raise ValueError(
                "COSMOS_DCP_PER_THREAD_COPY_AHEAD_BYTES must be auto, default, "
                f"or a non-negative integer; got {configured_copy_ahead!r}"
            ) from exc
        if copy_ahead_bytes < 0:
            raise ValueError(
                "COSMOS_DCP_PER_THREAD_COPY_AHEAD_BYTES must be non-negative; "
                f"got {copy_ahead_bytes}"
            )

    if copy_ahead_bytes is None:
        return FileSystemWriter(path=checkpoint_path)
    log.info(
        "DCP FileSystemWriter uses serial device-to-CPU staging "
        f"(per_thread_copy_ahead={copy_ahead_bytes}, "
        f"policy={configured_copy_ahead}, hip={torch.version.hip is not None})",
        rank0_only=False,
    )
    return FileSystemWriter(
        path=checkpoint_path,
        per_thread_copy_ahead=copy_ahead_bytes,
    )


def _checkpoint_async_with_pinned_memory(
    self, checkpoint_file: str, state_dict: Dict[str, Tuple[Any, str]]
) -> None:
    assert self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM, "Async mode must be AsyncMode.ASYNC_WITH_PINNED_MEM"

    from torch.distributed._state_dict_utils import _copy_state_dict, _create_cpu_state_dict

    if self.cpu_offload_state_dict is None:
        log.info(f"Preparing the CPU memory for staging")
        try:
            self.cpu_offload_state_dict = _create_cpu_state_dict(state_dict, pin_memory=True, share_memory=True)
        except Exception as e:
            log.warning(
                f"Failed to allocate shared memory for async DCP checkpoint ({e}). "
                "Falling back to synchronous checkpoint save."
            )
            self.async_mode = AsyncMode.DISABLED
            start_time = time.monotonic()
            self.save_state_dict_worker(state_dict, checkpoint_file)
            elapsed_time = time.monotonic() - start_time
            log.info(f"Fallback synchronous checkpoint save completed: Time taken: {elapsed_time:.2f} seconds")
            return

    log.info(f"Staging the state_dict in CPU memory")
    with torch.cuda.stream(self.staging_stream):
        self.cpu_offload_state_dict = _copy_state_dict(
            state_dict,
            self.cpu_offload_state_dict,
            non_blocking=True,
        )
        self.staging_ckpt_file = checkpoint_file

    self.staging_stream.synchronize()
    log.info(f"Staging the state_dict in CPU memory completed")

    self.mp_queue_send.put_nowait((self.cpu_offload_state_dict, self.staging_ckpt_file))
    self.checkpoint_in_progress = True
    log.info(f"Submitted checkpoint to background process")


def save_state_dict_worker(self, to_save_dict: Dict[str, Tuple[Any, str]], checkpoint_file: str) -> None:
    dcp_metadata_process_group = _get_dcp_metadata_process_group(self)
    for key, (v, full_checkpoint_path) in to_save_dict.items():
        if key == "dataloader":
            self._save_as_pkl(v, full_checkpoint_path)
        else:
            storage_writer = self.get_storage_writer(full_checkpoint_path)
            # Note that it is ok to create a new CustomSavePlanner object
            # for each checkpoint save since the save plans are cached in a
            # class dictionary.
            save_planner = CustomSavePlanner(
                dedup_save_to_lowest_rank=True,
                enable_plan_caching=True,
                cache_plans_key=f"custom_planner_{key}",
            )
            dcp.save(
                v,
                storage_writer=storage_writer,
                planner=save_planner,
                process_group=dcp_metadata_process_group,
            )

    if distributed.is_rank0():
        log.info(f"Saving last checkpoint file {checkpoint_file}")
        self._write_latest_checkpoint_file(checkpoint_file)

    log.info(f"Saved checkpoint to {os.path.join(self.save_dirname, checkpoint_file)}")


@misc.timer("checkpoint loading")
def load(
    self,
    model: ImaginaireModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> int:
    if self.callbacks is not None:
        self.callbacks.on_load_checkpoint_start(model)

    resume_keys, checkpoint_path, warm_start = self.keys_to_resume_during_load()
    resume_keys = sorted(resume_keys)
    log.critical(f"Resuming ckpt {checkpoint_path} with keys: {resume_keys}")

    iteration = 0
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    if checkpoint_path is not None:
        self._check_checkpoint_exists(checkpoint_path)

        for key in resume_keys:
            dist.barrier()

            cur_key_ckpt_full_path = os.path.join(checkpoint_path, key)
            log.critical(f"Start loading checkpoint from {cur_key_ckpt_full_path}")

            storage_reader = self.get_storage_reader(cur_key_ckpt_full_path)
            strict_resume = self.config_checkpoint.strict_resume

            # Note that we only allow skipping loading of keys during warm start. If the checkpoint is
            # the latest checkpoint of the same model, then we don't need to skip any keys.
            keys_to_skip_loading = self.config_checkpoint.keys_to_skip_loading if warm_start else []

            # Dedup-load context: when enabled, replicated state is read by a single rank and
            # broadcast over the device mesh instead of being read redundantly by every rank. The
            # per-tensor replicate group is derived from each DTensor's own mesh (see
            # _broadcast_state_dict). Applied only to the "model" and "optim" components
            # (the bulk of the data); the per-rank-unique "dataloader" state and
            # "trainer" RNG and the tiny "scheduler" state stay on the normal read-everywhere path.
            if key in ("model", "optim"):
                load_planner = CustomLoadPlanner(
                    allow_partial_load=not strict_resume,
                    keys_to_skip_loading=keys_to_skip_loading,
                    dedup=self.config_checkpoint.dcp_load_dedup,
                    global_rank=global_rank,
                )
            else:
                load_planner = dcp.DefaultLoadPlanner(allow_partial_load=not strict_resume)

            if key == "model":
                log.info("- Loading the model...")
                _model_wrapper = ModelWrapper(model)
                _state_dict = _model_wrapper.state_dict()
                dcp.load(
                    _state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                    no_dist=True,
                )
                if self.config_checkpoint.dcp_load_dedup:
                    # Fill in the reads the planner dropped by broadcasting from each leaf's
                    # single reader. Must run before the EMA copy and load_state_dict below.
                    _broadcast_state_dict(_state_dict, global_rank)
                if self.config_checkpoint.load_ema_to_reg and warm_start:
                    # The model has both net.* and net_ema.* submodules, so _state_dict
                    # contains both sets of keys after dcp.load(). Copy EMA weights into
                    # regular model weights so we can warm-start from EMA (and reset EMA
                    # when load_training_state=False).
                    #
                    # This must only run on warm start. During regular auto-resume
                    # (warm_start=False) the full training state is always reloaded, so
                    # copying EMA->reg here would silently overwrite the resumed regular
                    # weights with the (lagging) EMA snapshot on every restart while the
                    # optimizer state still tracks the pre-copy trajectory -- corrupting
                    # training in a way that depends on how often the job is preempted.
                    _copy_ema_weights_to_reg(_state_dict)

                results = _model_wrapper.load_state_dict(_state_dict)
                if len(results.missing_keys) > 0:
                    raise ValueError(f"Missing keys (not found in checkpoint): {results.missing_keys}")
                if len(results.unexpected_keys) > 0:
                    raise ValueError(
                        f"Unexpected keys (found in checkpoint but not in model): {results.unexpected_keys}"
                    )
                # Warm start that skipped net_ema (e.g. loading an EMA-only HF export
                # with no net_ema.* keys): the EMA shadow would otherwise keep its random
                # build-time generation pathway (init_moe is skipped when a checkpoint is
                # present). Seed net_ema from the freshly loaded net so the EMA starts equal
                # to net ("EMA warm-starts from net") instead of from random weights.
                if warm_start and any("net_ema" in skip_key for skip_key in keys_to_skip_loading):
                    ema_worker = getattr(model, "net_ema_worker", None)
                    if ema_worker is not None and getattr(model, "net_ema", None) is not None:
                        ema_worker.copy_to(src_model=model.net, tgt_model=model.net_ema)
                        log.info("Warm start: re-seeded net_ema from net (net_ema was skipped on load).")

            elif key == "optim":
                log.info("- Loading the optimizer...")
                _state_dict = optimizer.state_dict()
                dcp.load(
                    _state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                    no_dist=True,
                )
                if self.config_checkpoint.dcp_load_dedup:
                    # Fill in the reads the planner dropped by broadcasting from each leaf's
                    # single reader. Must run before load_state_dict below.
                    _broadcast_state_dict(_state_dict, global_rank)
                optimizer.load_state_dict(_state_dict)

            elif key == "scheduler":
                log.info("- Loading the scheduler...")
                _state_dict = scheduler.state_dict()
                dcp.load(
                    _state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                    no_dist=True,
                )
                scheduler.load_state_dict(_state_dict)

            elif key == "trainer":
                log.info("- Loading the trainer...")

                # Use rank-specific key for RNG state to support correct per-rank restoration
                rng_key = f"rng_state_{dist.get_rank()}"
                current_rng_state = get_rand_state_dict()
                _state_dict = {
                    "grad_scaler": grad_scaler.state_dict(),
                    "iteration": iteration,
                }
                # Check if rng_key exists in checkpoint metadata to avoid failure with strict_resume=True
                metadata = storage_reader.read_metadata()
                rng_key_exists = any(
                    k.startswith(f"{rng_key}.") or k == rng_key for k in metadata.state_dict_metadata.keys()
                )
                if rng_key_exists:
                    _state_dict[rng_key] = current_rng_state

                dcp.load(
                    _state_dict,
                    storage_reader=storage_reader,
                    planner=load_planner,
                    no_dist=True,
                )
                grad_scaler.load_state_dict(_state_dict["grad_scaler"])
                iteration = _state_dict["iteration"]
                set_rand_state_dict(_state_dict.get(rng_key, current_rng_state))

            elif key == "dataloader":
                if not easy_io.exists(cur_key_ckpt_full_path, backend_key=self.load_s3_backend_key):
                    log.info(
                        f"Checkpoint {cur_key_ckpt_full_path} does not exist, skip loading dataloader.",
                        rank0_only=False,
                    )
                    continue

                rank = dist.get_rank()
                dataloader_pkl_path = os.path.join(cur_key_ckpt_full_path, f"rank_{rank}.pkl")
                if not easy_io.exists(dataloader_pkl_path, backend_key=self.load_s3_backend_key):
                    log.info(f"No dataloader checkpoint found at {dataloader_pkl_path}", rank0_only=False)
                    continue

                log.info(f"- Loading the dataloader {cur_key_ckpt_full_path}...", rank0_only=False)
                _state_dict = easy_io.load(
                    dataloader_pkl_path,
                    file_format="pkl",
                    backend_key=self.load_s3_backend_key,
                )
                dataloader_wrapper = _DataloaderWrapper(self.callbacks)
                if dataloader_wrapper.has_state():
                    dataloader_wrapper.load_state_dict(_state_dict)

            else:
                raise ValueError(f"Invalid key: {key}. not support to resume.")

        if self.callbacks is not None and resume_keys:
            # Note that this callback is never used in the codebase.
            self.callbacks.on_load_checkpoint(model, state_dict={})
        log.info(f"Loaded checkpoint from {checkpoint_path} in iteration {iteration}")

    else:
        log.info("Training from scratch.")

    torch.cuda.empty_cache()

    if self.callbacks is not None:
        self.callbacks.on_load_checkpoint_end(model, iteration=iteration, checkpoint_path=checkpoint_path)
    return iteration


bind_missing_globals(globals(), _dcp_mod)
