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

"""PyTorch reference for MMCV multi-scale deformable attention tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_scale_deformable_attn_reference(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Evaluate multi-scale deformable attention with ``grid_sample``."""

    batch, _, num_heads, channels_per_head = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    level_shapes = value_spatial_shapes.detach().cpu().tolist()
    output = value.new_zeros(batch, num_queries, num_heads, channels_per_head)
    level_start = 0

    for level in range(num_levels):
        height, width = (int(dimension) for dimension in level_shapes[level])
        level_end = level_start + height * width
        feature_map = (
            value[:, level_start:level_end]
            .permute(0, 2, 3, 1)
            .reshape(batch * num_heads, channels_per_head, height, width)
        )
        grid = (
            sampling_locations[:, :, :, level]
            .permute(0, 2, 1, 3, 4)
            .reshape(batch * num_heads, num_queries, num_points, 2)
            .mul(2)
            .sub(1)
        )
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.reshape(
            batch, num_heads, channels_per_head, num_queries, num_points
        ).permute(0, 3, 1, 4, 2)
        level_weights = attention_weights[:, :, :, level].unsqueeze(-1)
        output = output + (sampled * level_weights).sum(dim=3)
        level_start = level_end

    return output.reshape(batch, num_queries, num_heads * channels_per_head)
