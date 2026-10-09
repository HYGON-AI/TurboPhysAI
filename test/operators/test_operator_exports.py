# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

import turbo_physai.operators as operators


EXPECTED_EXPORTS = {
    "DeformableAggregationFunction",
    "ModulatedDeformConv2dFunction",
    "bev_pool",
    "bev_pool_prepare",
    "bev_pool_prepare_geometry",
    "deformable_aggregation_function",
    "dynamic_scatter",
    "dynamic_voxelize",
    "grid_sample",
    "hard_voxelize",
    "indice_conv",
    "indice_maxpool",
    "interpolate",
    "modulated_deform_conv2d",
    "ms_deform_attn_backward",
    "ms_deform_attn_forward",
    "get_indice_pairs",
    "voxelize",
}


def test_operators_all_matches_public_high_level_api():
    assert set(operators.__all__) == EXPECTED_EXPORTS


@pytest.mark.parametrize(
    "name",
    [
        "get_indice_pairs_2d",
        "indice_conv_fp32",
        "indice_maxpool_half",
        "dynamic_point_to_voxel_forward",
        "bev_pool_forward",
    ],
)
def test_low_level_exports_are_not_public(name):
    assert name not in operators.__all__
    with pytest.raises(AttributeError):
        getattr(operators, name)
