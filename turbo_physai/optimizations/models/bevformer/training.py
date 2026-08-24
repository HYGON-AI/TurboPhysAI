# Copyright 2018-2019 OpenMMLab. All rights reserved.
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

"""BEVFormer training runtime recipe wrapper."""

import functools
import os
import os.path as osp


class _StopBenchmark(Exception):
    pass


def _profiler_hook_class():
    import torch
    from mmcv.runner import Hook, get_dist_info

    class IterRangeTorchProfilerHook(Hook):
        def __init__(self, start_iter=50, end_iter=60, log_dir=None,
                     stop_after=True):
            self.start_iter = int(start_iter)
            self.end_iter = int(end_iter)
            self.log_dir = log_dir
            self.stop_after = bool(stop_after)
            self.prof = None
            self._running = False
            self._rank = 0
            self._json_trace_path = None
            self._trace_exported = False

        def before_run(self, runner):
            if self.start_iter < 1 or self.end_iter < self.start_iter:
                runner.logger.warning(
                    "Skip torch profiler, invalid iter range: [%d, %d].",
                    self.start_iter, self.end_iter)
                return

            self._rank, _ = get_dist_info()
            log_dir = self.log_dir or osp.join(
                runner.work_dir, "profiler_logs")
            os.makedirs(log_dir, exist_ok=True)
            self._json_trace_path = osp.join(
                log_dir,
                f"trace_rank{self._rank}_iter{self.start_iter}_"
                f"{self.end_iter}.json")

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=False,
                with_stack=True)
            runner.logger.info(
                "Enable torch profiler for iter [%d, %d], rank=%d, "
                "output dir: %s",
                self.start_iter, self.end_iter, self._rank, log_dir)
            print(
                f"[Profiler][rank{self._rank}] armed for iter "
                f"[{self.start_iter}, {self.end_iter}] -> {log_dir}",
                flush=True)

        def after_train_iter(self, runner):
            if self.prof is None:
                return

            current_iter = runner.iter + 1
            if current_iter == self.start_iter and not self._running:
                self.prof.start()
                self._running = True
                runner.logger.info(
                    "torch profiler started at iter %d (rank=%d)",
                    current_iter, self._rank)
                print(
                    f"[Profiler][rank{self._rank}] started at iter "
                    f"{current_iter}", flush=True)

            if (self._running
                    and self.start_iter <= current_iter <= self.end_iter):
                self.prof.step()
                if self._rank == 0:
                    runner.logger.info(
                        "[Profiler] collecting iter %d/%d",
                        current_iter, self.end_iter)
                    print(
                        f"[Profiler] collecting iter "
                        f"{current_iter}/{self.end_iter}", flush=True)

            if current_iter == self.end_iter and self._running:
                self.prof.stop()
                self._running = False
                sort_key = "cuda_time_total"
                try:
                    profiler_table = self.prof.key_averages().table(
                        sort_by=sort_key, row_limit=20)
                except Exception:
                    sort_key = "cpu_time_total"
                    profiler_table = self.prof.key_averages().table(
                        sort_by=sort_key, row_limit=20)
                if profiler_table.strip():
                    runner.logger.info(
                        "torch profiler summary (rank=%d, sort_by=%s, "
                        "row_limit=20):\n%s",
                        self._rank, sort_key, profiler_table)
                    print(
                        f"[Profiler][rank{self._rank}] summary "
                        f"(sort_by={sort_key}, row_limit=20):\n"
                        f"{profiler_table}", flush=True)
                else:
                    event_count = len(self.prof.events())
                    runner.logger.warning(
                        "torch profiler summary is empty (rank=%d), "
                        "event_count=%d", self._rank, event_count)
                    print(
                        f"[Profiler][rank{self._rank}] summary is empty, "
                        f"event_count={event_count}", flush=True)
                self._export_trace_once(runner)
                runner.logger.info(
                    "torch profiler stopped at iter %d (rank=%d), json: %s",
                    current_iter, self._rank, self._json_trace_path)
                print(
                    f"[Profiler][rank{self._rank}] stopped at iter "
                    f"{current_iter}, json={self._json_trace_path}",
                    flush=True)
                if self.stop_after:
                    if self._rank == 0:
                        runner.logger.info(
                            "[Profiler] reached end iter %d, stopping "
                            "training by design.", current_iter)
                        print(
                            f"[Profiler] reached iter {current_iter}, "
                            "stopping training.", flush=True)
                    raise _StopBenchmark

        def after_run(self, runner):
            if self.prof is not None and self._running:
                self.prof.stop()
                self._running = False
                if self._json_trace_path is not None:
                    self._export_trace_once(runner)
                    runner.logger.info(
                        "torch profiler stopped in after_run (rank=%d), "
                        "json: %s", self._rank, self._json_trace_path)
                    print(
                        f"[Profiler][rank{self._rank}] stopped in "
                        f"after_run, json={self._json_trace_path}",
                        flush=True)
                else:
                    runner.logger.info("torch profiler stopped in after_run")

        def _export_trace_once(self, runner):
            if self._trace_exported or self._json_trace_path is None:
                return
            try:
                self.prof.export_chrome_trace(self._json_trace_path)
                self._trace_exported = True
            except RuntimeError as exc:
                if "Trace is already saved" in str(exc):
                    self._trace_exported = True
                    runner.logger.warning(
                        "Trace was already saved by profiler internals "
                        "(rank=%d).", self._rank)
                else:
                    raise

    return IterRangeTorchProfilerHook


def _max_train_iter_hook_class():
    from mmcv.runner import Hook

    class MaxTrainIterHook(Hook):
        def __init__(self, max_iters):
            self.max_iters = int(max_iters)

        def after_train_iter(self, runner):
            if runner.iter + 1 >= self.max_iters:
                runner.logger.info(
                    "Stop training after %d iterations for accuracy check.",
                    self.max_iters,
                )
                raise _StopBenchmark

    return MaxTrainIterHook


def training_runtime_wrapper(original, options):
    del options
    import torch

    @functools.wraps(original)
    def wrapped(model, dataset, cfg, *args, **kwargs):
        # The optimized BEVFormer entrypoint explicitly uses ``fork`` for
        # DataLoader workers.  Keep the component integration equivalent:
        # under ``spawn`` every worker re-imports the complete plugin and the
        # TurboPhysAI optimization catalog, adding tens of seconds to the first batch.
        dataloader_start_method = os.getenv(
            "TURBO_PHYSAI_DATALOADER_START_METHOD", "fork")
        if dataloader_start_method:
            current_start_method = torch.multiprocessing.get_start_method(
                allow_none=True)
            if current_start_method != dataloader_start_method:
                torch.multiprocessing.set_start_method(
                    dataloader_start_method,
                    force=current_start_method is not None,
                )

        optimizer = cfg.optimizer
        optimizer_type = (
            optimizer.get("type") if hasattr(optimizer, "get") else None
        )
        if optimizer_type != "AdamW":
            raise ValueError(
                "TurboPhysAI BEVFormer runtime_hcu requires optimizer.type=AdamW"
            )
        optimizer["fused"] = True
        cfg.data.samples_per_gpu = 2
        cfg.data.workers_per_gpu = int(
            os.getenv(
                "TURBO_PHYSAI_WORKERS_PER_GPU",
                str(cfg.data.get("workers_per_gpu", 8)),
            )
        )
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        model = model.to(memory_format=torch.channels_last)

        if os.getenv("ENABLE_TORCH_PROFILER", "0") == "1":
            profiler_log_dir = os.getenv("PROFILER_LOG_DIR", "") or None
            profiler_hook = dict(
                type=_profiler_hook_class(),
                start_iter=int(os.getenv("PROFILER_START_ITER", "50")),
                end_iter=int(os.getenv("PROFILER_END_ITER", "60")),
                log_dir=profiler_log_dir,
                stop_after=os.getenv("PROFILER_STOP_AFTER", "1") == "1",
                priority="LOW",
            )
            custom_hooks = cfg.get("custom_hooks", None)
            if custom_hooks is None:
                cfg.custom_hooks = []
            cfg.custom_hooks.append(profiler_hook)

        max_train_iters = cfg.get("max_train_iters", None)
        if max_train_iters:
            if cfg.get("custom_hooks", None) is None:
                cfg.custom_hooks = []
            cfg.custom_hooks.append(dict(
                type=_max_train_iter_hook_class(),
                max_iters=int(max_train_iters),
                priority="LOWEST",
            ))

        try:
            return original(model, dataset, cfg, *args, **kwargs)
        except _StopBenchmark:
            return None

    return wrapped
