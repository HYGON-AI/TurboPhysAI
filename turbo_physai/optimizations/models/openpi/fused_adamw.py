# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent fused AdamW replacement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def wrap_fused_adamw(original, options: Mapping[str, Any]):
    fused = bool(options.get("fused", True))

    def wrapped(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["fused"] = fused
        return original(*args, **kwargs)

    return wrapped
