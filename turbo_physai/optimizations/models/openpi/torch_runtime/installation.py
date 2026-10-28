# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared one-time installation lifecycle for OpenPI runtime libraries."""

from __future__ import annotations

from functools import wraps
import importlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable


_INSTALLED: set[str] = set()


def _prepare_runtime() -> None:
    os.environ["OPENPI_TORCH_KERNEL_MODE"] = "gate"
    os.environ.setdefault("LOCAL_RANK", os.environ.get("RANK", "0"))
    output = os.environ.get("OPENPI_TORCH_TIMING_OUTPUT")
    if not output:
        log_root = os.environ.get("OPENPI_LOG_DIR")
        base = Path(log_root) if log_root else Path(tempfile.gettempdir())
        output = str(base / "openpi_turbo_kernel_runtime")
        os.environ["OPENPI_TORCH_TIMING_OUTPUT"] = output
    Path(output).mkdir(parents=True, exist_ok=True)




