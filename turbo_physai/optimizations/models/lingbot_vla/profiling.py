# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import logging
import os

import torch

logger = logging.getLogger(__name__)


def upload_trace(wandb_project=None, wandb_name=None, trace_dir=None):
    logger.warning(
        "helper.upload_trace is not implemented upstream; trace already on disk at %s "
        "(wandb_project=%s, wandb_name=%s)",
        trace_dir,
        wandb_project,
        wandb_name,
    )


def _install_upload_trace_stub() -> None:
    try:
        helper = importlib.import_module("lingbotvla.utils.helper")
    except ImportError:
        return
    if not hasattr(helper, "upload_trace"):
        helper.upload_trace = upload_trace
        logger.info("installed a no-op lingbotvla.utils.helper.upload_trace")


def create_profiler(
    start_step: int,
    end_step: int,
    trace_dir: str,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
):
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    def handler_fn(p):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f"trace_rank{rank}.json")
        p.export_chrome_trace(path)
        logger.info("rank %s profiling trace saved to %s", rank, path)

    _install_upload_trace_stub()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info("build profiler schedule - wait: %s, warmup: %s, active: %s.", wait, warmup, active)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=True,
        with_stack=with_stack,
    )


_install_upload_trace_stub()
