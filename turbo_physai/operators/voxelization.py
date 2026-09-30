# Copyright 2018-2019 OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""MMDetection3D-compatible voxelization operators."""

from __future__ import annotations

import torch


def _ops():
    from turbo_physai import _C

    return _C


def dynamic_voxelize(points, voxel_size, coors_range, ndim=3):
    """Return one integer voxel coordinate for each input point."""

    coords = points.new_zeros((points.size(0), int(ndim)), dtype=torch.int)
    _ops().dynamic_voxelize(
        points, coords, voxel_size, coors_range, int(ndim)
    )
    return coords


def hard_voxelize(
    points,
    voxel_size,
    coors_range,
    max_points=35,
    max_voxels=20000,
    ndim=3,
    deterministic=True,
):
    """Group points into bounded voxels using the bundled native kernel."""

    max_points = int(max_points)
    max_voxels = int(max_voxels)
    voxels = points.new_zeros((max_voxels, max_points, points.size(1)))
    coords = points.new_zeros((max_voxels, int(ndim)), dtype=torch.int)
    point_counts = points.new_zeros((max_voxels,), dtype=torch.int)
    voxel_count = _ops().hard_voxelize(
        points,
        voxels,
        coords,
        point_counts,
        voxel_size,
        coors_range,
        max_points,
        max_voxels,
        int(ndim),
        bool(deterministic),
    )
    return (
        voxels[:voxel_count],
        coords[:voxel_count],
        point_counts[:voxel_count],
    )


def _dynamic_point_to_voxel_forward(feats, coors, reduce_type="max"):
    if reduce_type not in {"max", "sum", "mean"}:
        raise ValueError(f"unsupported reduce type: {reduce_type}")
    return _ops().dynamic_point_to_voxel_forward(
        feats.contiguous(), coors.int().contiguous(), reduce_type
    )


def _dynamic_point_to_voxel_backward(
    grad_feats,
    grad_reduced_feats,
    feats,
    reduced_feats,
    coors_idx,
    reduce_count,
    reduce_type="max",
):
    if reduce_type not in {"max", "sum", "mean"}:
        raise ValueError(f"unsupported reduce type: {reduce_type}")
    _ops().dynamic_point_to_voxel_backward(
        grad_feats,
        grad_reduced_feats.contiguous(),
        feats,
        reduced_feats,
        coors_idx,
        reduce_count,
        reduce_type,
    )


class _DynamicScatterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, coors, reduce_type="max"):
        reduced, out_coors, coors_idx, reduce_count = (
            _dynamic_point_to_voxel_forward(feats, coors, reduce_type)
        )
        ctx.save_for_backward(feats, reduced, coors_idx, reduce_count)
        ctx.reduce_type = reduce_type
        ctx.mark_non_differentiable(out_coors)
        return reduced, out_coors

    @staticmethod
    def backward(ctx, grad_reduced, grad_coors=None):
        del grad_coors
        feats, reduced, coors_idx, reduce_count = ctx.saved_tensors
        grad_feats = torch.empty_like(feats)
        _dynamic_point_to_voxel_backward(
            grad_feats,
            grad_reduced,
            feats,
            reduced,
            coors_idx,
            reduce_count,
            ctx.reduce_type,
        )
        return grad_feats, None, None


def dynamic_scatter(feats, coors, reduce_type="max"):
    """Reduce point features by coordinate, returning features and coordinates."""

    return _DynamicScatterFunction.apply(feats, coors, reduce_type)


def voxelize(
    points,
    voxel_size,
    coors_range,
    max_points=35,
    max_voxels=20000,
    deterministic=True,
):
    """Compatibility frontend matching MMDetection3D voxelization output."""

    if max_points == -1 or max_voxels == -1:
        return dynamic_voxelize(points, voxel_size, coors_range)
    return hard_voxelize(
        points,
        voxel_size,
        coors_range,
        max_points,
        max_voxels,
        3,
        deterministic,
    )


__all__ = [
    "dynamic_scatter",
    "dynamic_voxelize",
    "hard_voxelize",
    "voxelize",
]
