# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Configure DDP for OpenPI's compile stack."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def wrap_ddp_static_fast_path(original: Any, options: Mapping[str, Any]):
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        import torch

        mode = options.get("optimize_ddp", False)
        if isinstance(mode, str):
            normalized = mode.strip().lower()
            if normalized in {"0", "false", "off"}:
                mode = False
            elif normalized == "no_optimization":
                mode = "no_optimization"
            elif normalized in {"1", "true", "on", "default"}:
                mode = True
        torch._dynamo.config.optimize_ddp = mode

        call_kwargs = dict(kwargs)
        call_kwargs.update(
            broadcast_buffers=_bool(options.get("broadcast_buffers"), False),
            find_unused_parameters=_bool(options.get("find_unused_parameters"), False),
            gradient_as_bucket_view=_bool(options.get("gradient_as_bucket_view"), True),
            static_graph=_bool(options.get("static_graph"), True),
        )
        return original(self, *args, **call_kwargs)

    return wrapped
