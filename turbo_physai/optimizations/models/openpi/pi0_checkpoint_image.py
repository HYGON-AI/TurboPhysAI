# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""External replacement for the official PI0 outer checkpoint dispatcher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .pi0_checkpoint_state import register_image


def wrap_pi0_image_apply_checkpoint(original, options: Mapping[str, Any]):
    """Register image-region options and return the shared dispatcher."""

    return register_image(original, options)
