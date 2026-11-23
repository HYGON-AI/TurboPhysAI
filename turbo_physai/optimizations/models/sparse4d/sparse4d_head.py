# Derived from Sparse4D: projects/mmdet3d_plugin/models/sparse4d_head.py;
# projects/mmdet3d_plugin/models/blocks.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

"""Sparse4DHead Optimization.

Leave instance IDs on the CPU for samplers that batch the device transfer.
"""

from typing import List, Union

import torch


def forward(self, feature_maps: Union[torch.Tensor, List], metas: dict):
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = [feature_maps]
    batch_size = feature_maps[0].shape[0]

    # ========= get instance info ============
    if (
        self.sampler.dn_metas is not None
        and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
    ):
        self.sampler.dn_metas = None
    (
        instance_feature,
        anchor,
        temp_instance_feature,
        temp_anchor,
        time_interval,
    ) = self.instance_bank.get(
        batch_size, metas, dn_metas=self.sampler.dn_metas
    )

    # ========= prepare for denosing training ============
    # 1. get dn metas: noisy-anchors and corresponding GT
    # 2. concat learnable instances and noisy instances
    # 3. get attention mask
    attn_mask = None
    dn_metas = None
    temp_dn_reg_target = None
    if self.training and hasattr(self.sampler, "get_dn_anchors"):
        if "instance_id" in metas["img_metas"][0]:
            gt_instance_id = [
                torch.from_numpy(x["instance_id"])
                for x in metas["img_metas"]
            ]
        else:
            gt_instance_id = None
        dn_metas = self.sampler.get_dn_anchors(
            metas[self.gt_cls_key],
            metas[self.gt_reg_key],
            gt_instance_id,
        )
    if dn_metas is not None:
        (
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            dn_attn_mask,
            valid_mask,
            dn_id_target,
        ) = dn_metas
        num_dn_anchor = dn_anchor.shape[1]
        if dn_anchor.shape[-1] != anchor.shape[-1]:
            remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
            dn_anchor = torch.cat(
                [
                    dn_anchor,
                    dn_anchor.new_zeros(
                        batch_size, num_dn_anchor, remain_state_dims
                    ),
                ],
                dim=-1,
            )
        anchor = torch.cat([anchor, dn_anchor], dim=1)
        instance_feature = torch.cat(
            [
                instance_feature,
                instance_feature.new_zeros(
                    batch_size, num_dn_anchor, instance_feature.shape[-1]
                ),
            ],
            dim=1,
        )
        num_instance = instance_feature.shape[1]
        num_free_instance = num_instance - num_dn_anchor
        attn_mask = anchor.new_ones(
            (num_instance, num_instance), dtype=torch.bool
        )
        attn_mask[:num_free_instance, :num_free_instance] = False
        attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

    anchor_embed = self.anchor_encoder(anchor)
    if temp_anchor is not None:
        temp_anchor_embed = self.anchor_encoder(temp_anchor)
    else:
        temp_anchor_embed = None

    # =================== forward the layers ====================
    prediction = []
    classification = []
    quality = []
    for i, op in enumerate(self.operation_order):
        if self.layers[i] is None:
            continue
        elif op == "temp_gnn":
            instance_feature = self.graph_model(
                i,
                instance_feature,
                temp_instance_feature,
                temp_instance_feature,
                query_pos=anchor_embed,
                key_pos=temp_anchor_embed,
                attn_mask=attn_mask
                if temp_instance_feature is None
                else None,
            )
        elif op == "gnn":
            instance_feature = self.graph_model(
                i,
                instance_feature,
                value=instance_feature,
                query_pos=anchor_embed,
                attn_mask=attn_mask,
            )
        elif op == "norm" or op == "ffn":
            instance_feature = self.layers[i](instance_feature)
        elif op == "deformable":
            instance_feature = self.layers[i](
                instance_feature,
                anchor,
                anchor_embed,
                feature_maps,
                metas,
            )
        elif op == "refine":
            anchor, cls, qt = self.layers[i](
                instance_feature,
                anchor,
                anchor_embed,
                time_interval=time_interval,
                return_cls=(
                    self.training
                    or len(prediction) == self.num_single_frame_decoder - 1
                    or i == len(self.operation_order) - 1
                ),
            )
            prediction.append(anchor)
            classification.append(cls)
            quality.append(qt)
            if len(prediction) == self.num_single_frame_decoder:
                instance_feature, anchor = self.instance_bank.update(
                    instance_feature, anchor, cls
                )
                if (
                    dn_metas is not None
                    and self.sampler.num_temp_dn_groups > 0
                    and dn_id_target is not None
                ):
                    (
                        instance_feature,
                        anchor,
                        temp_dn_reg_target,
                        temp_dn_cls_target,
                        temp_valid_mask,
                        dn_id_target,
                    ) = self.sampler.update_dn(
                        instance_feature,
                        anchor,
                        dn_reg_target,
                        dn_cls_target,
                        valid_mask,
                        dn_id_target,
                        self.instance_bank.num_anchor,
                        self.instance_bank.mask,
                    )
            if i != len(self.operation_order) - 1:
                anchor_embed = self.anchor_encoder(anchor)
            if (
                len(prediction) > self.num_single_frame_decoder
                and temp_anchor_embed is not None
            ):
                temp_anchor_embed = anchor_embed[
                    :, : self.instance_bank.num_temp_instances
                ]
        else:
            raise NotImplementedError(f"{op} is not supported.")

    output = {}

    # split predictions of learnable instances and noisy instances
    if dn_metas is not None:
        dn_classification = [
            x[:, num_free_instance:] for x in classification
        ]
        classification = [x[:, :num_free_instance] for x in classification]
        dn_prediction = [x[:, num_free_instance:] for x in prediction]
        prediction = [x[:, :num_free_instance] for x in prediction]
        quality = [
            x[:, :num_free_instance] if x is not None else None
            for x in quality
        ]
        output.update(
            {
                "dn_prediction": dn_prediction,
                "dn_classification": dn_classification,
                "dn_reg_target": dn_reg_target,
                "dn_cls_target": dn_cls_target,
                "dn_valid_mask": valid_mask,
            }
        )
        if temp_dn_reg_target is not None:
            output.update(
                {
                    "temp_dn_reg_target": temp_dn_reg_target,
                    "temp_dn_cls_target": temp_dn_cls_target,
                    "temp_dn_valid_mask": temp_valid_mask,
                    "dn_id_target": dn_id_target,
                }
            )
            dn_cls_target = temp_dn_cls_target
            valid_mask = temp_valid_mask
        dn_instance_feature = instance_feature[:, num_free_instance:]
        dn_anchor = anchor[:, num_free_instance:]
        instance_feature = instance_feature[:, :num_free_instance]
        anchor = anchor[:, :num_free_instance]
        cls = cls[:, :num_free_instance]

        # cache dn_metas for temporal denoising
        self.sampler.cache_dn(
            dn_instance_feature,
            dn_anchor,
            dn_cls_target,
            valid_mask,
            dn_id_target,
        )
    output.update(
        {
            "classification": classification,
            "prediction": prediction,
            "quality": quality,
        }
    )

    # cache current instances for temporal modeling
    self.instance_bank.cache(
        instance_feature, anchor, cls, metas, feature_maps
    )
    if not self.training:
        instance_id = self.instance_bank.get_instance_id(
            cls, anchor, self.decoder.score_threshold
        )
        output["instance_id"] = instance_id
    return output


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
