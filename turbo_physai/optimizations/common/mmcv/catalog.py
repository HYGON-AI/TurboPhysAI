# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python declarations for reusable MMCV optimizations."""

from __future__ import annotations

from ....engine.definitions import group, replace


MSDA = group(
    "mmcv.msda",
    replace(
        target="mmcv._ext.ms_deform_attn_forward",
        replacement=(
            "turbo_physai.optimizations.common.mmcv.msda."
            "ms_deform_attn_forward"
        ),
    ),
    replace(
        target="mmcv._ext.ms_deform_attn_backward",
        replacement=(
            "turbo_physai.optimizations.common.mmcv.msda."
            "ms_deform_attn_backward"
        ),
    ),
)


__all__ = ["MSDA"]
