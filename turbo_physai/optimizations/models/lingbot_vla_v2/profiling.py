# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""``lingbotvla.utils.helper.create_profiler`` 的替代版：调度与上游一致，另补上仓库缺失的 upload_trace。"""

from __future__ import annotations

import importlib
import logging

import torch

logger = logging.getLogger(__name__)


def upload_trace(wandb_project=None, wandb_name=None, trace_dir=None):
    """上游调用但从未定义的占位实现：trace 已落盘，这里只记日志。"""
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
    start_step: int, end_step: int, trace_dir: str, record_shapes: bool, profile_memory: bool, with_stack: bool
):
    def handler_fn(p):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        torch.profiler.tensorboard_trace_handler(trace_dir)(p)
        logger.info(f"Profiling result saved at {trace_dir}.")

    _install_upload_trace_stub()

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    warmup = 0 if start_step == 1 else 1
    wait = start_step - warmup - 1
    active = end_step - start_step
    logger.info(f"build profiler schedule - wait: {wait}, warmup: {warmup}, active: {active}.")
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=handler_fn,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_modules=True,
        with_stack=with_stack,
    )
