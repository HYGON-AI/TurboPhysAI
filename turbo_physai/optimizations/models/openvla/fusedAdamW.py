# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def adamw_fused_wrapper(original: Any, options: Mapping[str, Any]):
    def adamw_factory(*args: Any, **kwargs: Any):
        kwargs.setdefault("fused", True)
        return original(*args, **kwargs)

    return adamw_factory