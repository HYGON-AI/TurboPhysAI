# Portions are adapted from the PointPillars HCU optimization project:
# Source: PointPillars/mmdetection3d/mmdet3d/ops/voxelize_opt/__init__.py
# Revision: bd37518d0408bcea4ac879a90256ffb8b8628258
# Project license: Apache-2.0
# Project copyright: Copyright 2018-2019 Open-MMLab. All rights reserved.
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic tensor hard-voxelization replacement."""

from functools import wraps
import os

import torch


def hard_voxelize_legacy_sorted(points, voxel_size, coors_range,
                                max_points, max_voxels):
    """Match optimized hard-voxel ordering and return its x-y-z coordinates."""
    if points.numel() == 0:
        return (points.new_zeros((0, max_points, points.shape[1])),
                torch.empty((0, 3), dtype=torch.int32, device=points.device),
                torch.empty((0,), dtype=torch.int32, device=points.device))

    vx, vy, vz = [float(v) for v in voxel_size]
    x_min, y_min, z_min, x_max, y_max, z_max = [
        float(v) for v in coors_range
    ]
    grid_x = round((x_max - x_min) / vx)
    grid_y = round((y_max - y_min) / vy)
    grid_z = round((z_max - z_min) / vz)

    cx = torch.floor((points[:, 0] - x_min) / vx).to(torch.int64)
    cy = torch.floor((points[:, 1] - y_min) / vy).to(torch.int64)
    cz = torch.floor((points[:, 2] - z_min) / vz).to(torch.int64)
    valid = ((cx >= 0) & (cx < grid_x) & (cy >= 0) & (cy < grid_y)
             & (cz >= 0) & (cz < grid_z))
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    if valid_idx.numel() == 0:
        return (points.new_zeros((0, max_points, points.shape[1])),
                torch.empty((0, 3), dtype=torch.int32, device=points.device),
                torch.empty((0,), dtype=torch.int32, device=points.device))

    keys = cx * (grid_y * grid_z) + cy * grid_z + cz
    order = torch.argsort(keys[valid_idx], stable=True)
    keys_sorted = keys[valid_idx][order]
    idx_sorted = valid_idx[order]

    new_group = torch.empty_like(keys_sorted, dtype=torch.bool)
    new_group[0] = True
    if keys_sorted.numel() > 1:
        new_group[1:] = keys_sorted[1:] != keys_sorted[:-1]
    group_starts = new_group.nonzero(as_tuple=False).flatten()
    num_groups = int(group_starts.numel())
    group_ids = torch.cumsum(new_group.to(torch.int64), dim=0) - 1
    sorted_pos = torch.arange(keys_sorted.numel(), device=points.device)
    rank_sorted = sorted_pos - group_starts[group_ids]

    group_ends = torch.cat((
        group_starts[1:],
        torch.tensor([keys_sorted.numel()], device=points.device,
                     dtype=torch.int64),
    ))
    group_counts = group_ends - group_starts
    first_indices = idx_sorted[group_starts]
    group_order = torch.argsort(first_indices, stable=True)
    voxel_id_by_group = torch.empty(
        (num_groups,), device=points.device, dtype=torch.int64
    )
    voxel_id_by_group[group_order] = torch.arange(
        num_groups, device=points.device, dtype=torch.int64
    )
    voxel_id_sorted = voxel_id_by_group[group_ids]

    num_out = min(num_groups, max_voxels)
    voxels = points.new_zeros((num_out, max_points, points.shape[1]))
    keep = (voxel_id_sorted < max_voxels) & (rank_sorted < max_points)
    if bool(keep.any()):
        voxels[voxel_id_sorted[keep], rank_sorted[keep]] = points[
            idx_sorted[keep]
        ]

    selected_groups = group_order[:num_out]
    coors_xyz = torch.stack((
        cx[idx_sorted[group_starts]],
        cy[idx_sorted[group_starts]],
        cz[idx_sorted[group_starts]],
    ), dim=1)[selected_groups]
    coors_xyz = coors_xyz.to(torch.int32)
    counts = torch.minimum(
        group_counts[selected_groups],
        torch.full((num_out,), max_points, device=points.device,
                   dtype=torch.int64),
    ).to(torch.int32)
    return voxels, coors_xyz, counts


def _can_use_legacy_sorted(self, points, max_voxels):
    mode = os.getenv("MMDET3D_VOXELIZE_OPT_MODE", "legacy").lower()
    disabled = os.getenv("MMDET3D_DISABLE_VOXELIZE_OPT", "0") == "1"
    return (
        not disabled
        and mode in ("legacy", "safe", "1", "true", "on")
        and self.deterministic
        and points.is_cuda
        and getattr(torch.version, "hip", None) is not None
        and points.dtype == torch.float32
        and points.ndim == 2
        and points.shape[1] >= 3
        and self.max_num_points > 0
        and max_voxels > 0
        and len(self.voxel_size) == 3
    )


def supports_legacy_sorted(self, input):
    """Select only the deterministic hard-voxel path tested for this model."""
    if not isinstance(input, torch.Tensor):
        return False
    max_voxels = self.max_voxels[0 if self.training else 1]
    return bool(_can_use_legacy_sorted(self, input, max_voxels))


def voxelization_layer_forward_wrapper(original, options):
    """Replace VoxelizationByGridShape.forward on the supported HCU path."""
    @wraps(original)
    def wrapped(self, input):
        max_voxels = self.max_voxels[0 if self.training else 1]
        if not _can_use_legacy_sorted(self, input, max_voxels):
            return original(self, input)
        voxels, coors_xyz, counts = hard_voxelize_legacy_sorted(
            input.contiguous(), self.voxel_size, self.point_cloud_range,
            self.max_num_points, max_voxels,
        )
        return voxels, coors_xyz[:, [2, 1, 0]], counts

    return wrapped
