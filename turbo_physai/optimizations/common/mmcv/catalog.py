# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python declarations for reusable MMCV optimizations."""

from __future__ import annotations

from ....engine.definitions import group, replace


MDC = group(
    "mmcv.mdc",
    replace(
        target="mmcv.ops.modulated_deform_conv.modulated_deform_conv2d",
        aliases=("mmcv.ops.modulated_deform_conv2d",),
        replacement=(
            "turbo_physai.optimizations.common.mmcv.modulated_deform_conv."
            "modulated_deform_conv2d"
        ),
        runtime_condition=(
            "turbo_physai.optimizations.common.mmcv.modulated_deform_conv."
            "is_supported_mdc_call"
        ),
    ),
)


MSDA = group(
    "mmcv.msda",
    replace(
        target="mmcv._ext.ms_deform_attn_forward",
        replacement=(
            "turbo_physai.operators.multi_scale_deformable_attention."
            "ms_deform_attn_forward"
        ),
    ),
    replace(
        target="mmcv._ext.ms_deform_attn_backward",
        replacement=(
            "turbo_physai.operators.multi_scale_deformable_attention."
            "ms_deform_attn_backward"
        ),
    ),
)


__all__ = ["MDC", "MSDA"]
