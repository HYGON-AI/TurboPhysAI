# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""Bounded GC timing diagnostics; never changes collection policy or CUDA state."""

import gc
import json
import logging
import os
import time
from collections import deque

logger = logging.getLogger(__name__)


class GCTrace:
    def __init__(self, output_path, rank):
        self.path = os.path.join(output_path, "gc_traces", f"rank{rank}.jsonl")
        self.rank = rank
        self.records = deque(maxlen=4096)
        self.callback = self._record
        self.file = None

    def _record(self, phase, info):
        # No logging, profiler ranges, I/O, tensor operations or collection here.
        self.records.append(
            (
                time.perf_counter_ns(),
                phase,
                info["generation"],
                info.get("collected", 0),
                info.get("uncollectable", 0),
            )
        )

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = open(self.path, "a", encoding="utf-8")
        self.file.write(
            json.dumps(
                {
                    "rank": self.rank,
                    "event": "clock_anchor",
                    "perf_counter_ns": time.perf_counter_ns(),
                    "unix_ns": time.time_ns(),
                    "pid": os.getpid(),
                    "capacity": self.records.maxlen,
                }
            )
            + "\n"
        )
        self.file.flush()
        gc.callbacks.append(self.callback)
        return self

    def step(self, completed_step):
        # Draining and writing occur at step boundaries, outside the callback.
        count = len(self.records)
        rows = [self.records.popleft() for _ in range(count)]
        self.file.write(
            json.dumps(
                {
                    "rank": self.rank,
                    "completed_step": completed_step,
                    "possibly_truncated": count == self.records.maxlen,
                    "events": rows,
                }
            )
            + "\n"
        )
        self.file.flush()

    def __exit__(self, *exc):
        gc.callbacks.remove(self.callback)
        try:
            self.step(None)
        except OSError:
            if exc[0] is None:
                raise
            logger.exception("Failed to flush GC diagnostics during training failure")
        finally:
            self.file.close()
