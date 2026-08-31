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

"""Complete optimized TransFusionBBoxCoder implementation."""

import torch
from torch import nn

from mmdet.core.bbox import BaseBBoxCoder


class TransFusionBBoxCoder(BaseBBoxCoder, nn.Module):
    """TransFusion coder with device-aware geometry constants."""

    def __init__(
        self,
        pc_range,
        out_size_factor,
        voxel_size,
        post_center_range=None,
        score_threshold=None,
        code_size=8,
    ):
        nn.Module.__init__(self)
        BaseBBoxCoder.__init__(self)
        self.out_size_factor = out_size_factor
        self.post_center_range = post_center_range
        self.score_threshold = score_threshold
        self.code_size = code_size
        self.register_buffer(
            "pc_range", torch.tensor(pc_range, dtype=torch.float32)
        )
        self.register_buffer(
            "voxel_size", torch.tensor(voxel_size, dtype=torch.float32)
        )

    def encode(self, dst_boxes):
        targets = torch.zeros(
            [dst_boxes.shape[0], self.code_size], device=dst_boxes.device
        )
        targets[:, 0] = (dst_boxes[:, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0]
        )
        targets[:, 1] = (dst_boxes[:, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1]
        )
        targets[:, 3] = dst_boxes[:, 3].log()
        targets[:, 4] = dst_boxes[:, 4].log()
        targets[:, 5] = dst_boxes[:, 5].log()
        targets[:, 2] = dst_boxes[:, 2] + dst_boxes[:, 5] * 0.5
        targets[:, 6] = torch.sin(dst_boxes[:, 6])
        targets[:, 7] = torch.cos(dst_boxes[:, 6])
        if self.code_size == 10:
            targets[:, 8:10] = dst_boxes[:, 7:]
        return targets

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        final_preds = heatmap.max(1, keepdims=False).indices
        final_scores = heatmap.max(1, keepdims=False).values

        center_new = (
            center * self.out_size_factor * self.voxel_size[:, None]
            + self.pc_range[:, None]
        )
        dim_new = dim.exp()
        height_new = height - dim_new[:, 2:3, :] * 0.5
        rotation = torch.atan2(rot[:, 0:1, :], rot[:, 1:2, :])
        parts = [center_new, height_new, dim_new, rotation]
        if vel is not None:
            parts.append(vel)
        final_box_preds = torch.cat(parts, dim=1).permute(0, 2, 1)

        predictions = [
            {
                "bboxes": final_box_preds[index],
                "scores": final_scores[index],
                "labels": final_preds[index],
            }
            for index in range(heatmap.shape[0])
        ]
        if filter is False:
            return predictions

        if self.score_threshold is not None:
            threshold_mask = final_scores > self.score_threshold
        if self.post_center_range is None:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only support "
                "post_center_range is not None for now!"
            )

        self.post_center_range = torch.tensor(
            self.post_center_range, device=heatmap.device
        )
        mask = (
            final_box_preds[..., :3] >= self.post_center_range[:3]
        ).all(2)
        mask &= (
            final_box_preds[..., :3] <= self.post_center_range[3:]
        ).all(2)
        predictions = []
        for index in range(heatmap.shape[0]):
            current_mask = mask[index, :]
            if self.score_threshold:
                current_mask &= threshold_mask[index]
            predictions.append(
                {
                    "bboxes": final_box_preds[index, current_mask],
                    "scores": final_scores[index, current_mask],
                    "labels": final_preds[index, current_mask],
                }
            )
        return predictions
