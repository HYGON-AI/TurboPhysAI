# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Public operator APIs, loaded on demand with their backend dependencies."""

from importlib import import_module
from typing import Any


_OPERATORS = {
    "bev_pool": ("turbo_physai.operators._bev_pool", "bev_pool"),
    "bev_pool_prepare": ("turbo_physai.operators._bev_pool", "bev_pool_prepare"),
    "bev_pool_prepare_geometry": (
        "turbo_physai.operators._bev_pool", "bev_pool_prepare_geometry",
    ),
    "get_indice_pairs": (
        "turbo_physai.operators.sparse_conv", "get_indice_pairs",
    ),
    "indice_conv": ("turbo_physai.operators.sparse_conv", "indice_conv"),
    "indice_maxpool": (
        "turbo_physai.operators.sparse_conv", "indice_maxpool",
    ),
    "ModulatedDeformConv2dFunction": (
        "turbo_physai.operators.modulated_deform_conv",
        "ModulatedDeformConv2dFunction",
    ),
    "modulated_deform_conv2d": (
        "turbo_physai.operators.modulated_deform_conv",
        "modulated_deform_conv2d",
    ),
    "grid_sample": ("turbo_physai.operators._grid_sample", "grid_sample"),
    "dynamic_scatter": (
        "turbo_physai.operators.voxelization", "dynamic_scatter",
    ),
    "dynamic_voxelize": (
        "turbo_physai.operators.voxelization", "dynamic_voxelize",
    ),
    "hard_voxelize": (
        "turbo_physai.operators.voxelization", "hard_voxelize",
    ),
    "voxelize": ("turbo_physai.operators.voxelization", "voxelize"),
    "interpolate": ("turbo_physai.operators.upsample_bilinear_2d", "interpolate"),
    "deformable_aggregation_function": (
        "turbo_physai.operators.deformable_aggregation",
        "deformable_aggregation_function",
    ),
    "DeformableAggregationFunction": (
        "turbo_physai.operators.deformable_aggregation",
        "DeformableAggregationFunction",
    ),
    "ms_deform_attn_forward": (
        "turbo_physai.operators.multi_scale_deformable_attention",
        "ms_deform_attn_forward",
    ),
    "ms_deform_attn_backward": (
        "turbo_physai.operators.multi_scale_deformable_attention",
        "ms_deform_attn_backward",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _OPERATORS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_OPERATORS)
