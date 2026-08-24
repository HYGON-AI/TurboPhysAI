# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python declarations for reusable MMDetection3D optimizations."""

from __future__ import annotations

from ....engine.definitions import group, replace

GAUSSIAN = group(
    "mmdet3d.gaussian",
    replace(
        target="mmdet3d.core.utils.gaussian.gaussian_2d",
        replacement="turbo_physai.optimizations.common.mmdet3d.gaussian.gaussian_2d",
    ),
)

BEV_POOL = group(
    "mmdet3d.bev_pool",
    replace(
        target="mmdet3d.ops.bev_pool.bev_pool.bev_pool",
        aliases=("mmdet3d.ops.bev_pool.bev_pool",),
        replacement="turbo_physai.optimizations.common.mmdet3d.bev_pool.bev_pool",
    ),
)

QUICK_CUMSUM = group(
    "mmdet3d.quick_cumsum",
    replace(
        target="mmdet3d.ops.bev_pool.bev_pool.QuickCumsum.forward",
        replacement=(
            "turbo_physai.optimizations.common.mmdet3d.bev_pool."
            "quick_cumsum_forward"
        ),
    ),
    replace(
        target="mmdet3d.ops.bev_pool.bev_pool.QuickCumsum.backward",
        replacement=(
            "turbo_physai.optimizations.common.mmdet3d.bev_pool."
            "quick_cumsum_backward"
        ),
    ),
)

VOXELIZATION = group(
    "mmdet3d.voxelization",
    replace(
        target="mmdet3d.ops.voxel.voxelize._Voxelization.forward",
        replacement=(
            "turbo_physai.optimizations.common.mmdet3d.voxelization."
            "voxelization_forward"
        ),
    ),
)

CANONICAL_INDICE_PAIRS = group(
    "mmdet3d.canonical_indice_pairs",
    replace(
        target="mmdet3d.ops.spconv.ops.get_indice_pairs",
        replacement=(
            "turbo_physai.optimizations.common.mmdet3d.sparse_conv."
            "get_indice_pairs"
        ),
    ),
)

SPARSE_TENSOR = group(
    "mmdet3d.sparse_tensor",
    replace(
        target="mmdet3d.ops.spconv.structure.SparseConvTensor.sparity",
        replacement="turbo_physai.optimizations.common.mmdet3d.sparse_tensor.sparity",
    ),
)


__all__ = [
    "BEV_POOL",
    "CANONICAL_INDICE_PAIRS",
    "GAUSSIAN",
    "QUICK_CUMSUM",
    "SPARSE_TENSOR",
    "VOXELIZATION",
]
