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

"""MMDetection3D voxelization frontend backed by the bundled extension."""


def _ops():
    from turbo_physai import ops

    return ops


def voxelization_forward(
    ctx,
    points,
    voxel_size,
    coors_range,
    max_points=35,
    max_voxels=20000,
    deterministic=True,
):
    del ctx
    import torch

    extension = _ops()
    if max_points == -1 or max_voxels == -1:
        coords = points.new_zeros((points.size(0), 3), dtype=torch.int)
        extension.dynamic_voxelize(points, coords, voxel_size, coors_range, 3)
        return coords

    voxels = points.new_zeros((max_voxels, max_points, points.size(1)))
    coords = points.new_zeros((max_voxels, 3), dtype=torch.int)
    point_counts = points.new_zeros((max_voxels,), dtype=torch.int)
    voxel_count = extension.hard_voxelize(
        points,
        voxels,
        coords,
        point_counts,
        voxel_size,
        coors_range,
        max_points,
        max_voxels,
        3,
        deterministic,
    )
    return (
        voxels[:voxel_count],
        coords[:voxel_count],
        point_counts[:voxel_count],
    )
