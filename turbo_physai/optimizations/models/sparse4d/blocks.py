# Derived from Sparse4D: projects/mmdet3d_plugin/models/blocks.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

import torch


def _get_weights(self, instance_feature, anchor_embed, metas=None):
    bs, num_anchor = instance_feature.shape[:2]
    feature = instance_feature + anchor_embed
    if self.camera_encoder is not None:
        camera_embed = self.camera_encoder(
            metas["projection_mat"][:, :, :3].reshape(
                bs, self.num_cams, -1
            )
        )
        feature = feature[:, :, None] + camera_embed[:, None]

    weights = (
        self.weights_fc(feature)
        .reshape(bs, num_anchor, -1, self.num_groups)
        .softmax(dim=-2)
        .reshape(
            bs,
            num_anchor,
            self.num_cams,
            self.num_levels,
            self.num_pts,
            self.num_groups,
        )
    )
    if self.training and self.attn_drop > 0:
        mask = torch.rand(
            bs, num_anchor, self.num_cams, 1, self.num_pts, 1,
        device=weights.device,dtype=weights.dtype)
        weights = ((mask > self.attn_drop) * weights) / (
            1 - self.attn_drop
        )
    return weights