# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""External replacement for the official PI0 outer checkpoint dispatcher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pi0_checkpoint_state import register_main


def wrap_pi0_main_apply_checkpoint(original, options: Mapping[str, Any]):
    """Register main-region options and return the shared dispatcher."""

    return register_main(original, options)
