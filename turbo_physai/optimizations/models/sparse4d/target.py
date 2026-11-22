# Derived from Sparse4D: projects/mmdet3d_plugin/models/detection3d/target.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

"""SparseBox3DTarget replacement installed in mmdet's BBOX_SAMPLERS registry."""

import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from torch_linear_assignment import batch_linear_assignment as _batch_linear_assignment

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.base_target import BaseTargetWithDenoising


__all__ = ["SparseBox3DTarget"]


@torch.library.custom_op(
    "turbo_physai::batch_linear_assignment", mutates_args=(), device_types=("cpu", "cuda")
)
def _batch_linear_assignment_op(cost: torch.Tensor) -> torch.Tensor:
    return _batch_linear_assignment(cost.detach())


@_batch_linear_assignment_op.register_fake
def _batch_linear_assignment_fake(cost: torch.Tensor) -> torch.Tensor:
    torch._check(cost.ndim == 3, lambda: "cost must have shape (batch, workers, tasks)")
    # The solver returns one index per worker, including unmatched workers.
    return cost.new_empty(cost.shape[:2], dtype=torch.long)


class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
    ):
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

    def encode_reg_target(self, box_target, device=None):
        lengths = [b.shape[0] for b in box_target]
        total = sum(lengths)
        if total == 0:
            out = [b.new_empty((0, 10)) for b in box_target]
            return [o.to(device) for o in out] if device is not None else out

        flat = torch.cat(box_target, dim=0)  # [sum(num_gt), 9]
        if device is not None and flat.device != device:
            flat = flat.to(device=device, non_blocking=True)

        encoded = self._encode_flat(flat)
        return list(encoded.split(lengths, dim=0))

    def _encode_flat(self, box):
        return torch.cat(
            [
                box[..., [X, Y, Z]],
                box[..., [W, L, H]].log(),
                torch.sin(box[..., YAW]).unsqueeze(-1),
                torch.cos(box[..., YAW]).unsqueeze(-1),
                box[..., YAW + 1:],
            ],
            dim=-1,
        )

    def encode_reg_target_fix(self, box_target, valid_mask):

        valid_mask = valid_mask.to(dtype=torch.bool)
        valid_mask_3d = valid_mask.unsqueeze(-1)

        box_size = box_target[..., [W, L, H]]
        safe_box_size = torch.where(
            valid_mask_3d,
            box_size,
            torch.ones_like(box_size),
        )

        output = torch.cat(
            [
                box_target[..., [X, Y, Z]],
                safe_box_size.log(),
                torch.sin(box_target[..., YAW]).unsqueeze(-1),
                torch.cos(box_target[..., YAW]).unsqueeze(-1),
                box_target[..., YAW + 1 :],
            ],
            dim=-1,
        )
        return torch.where(valid_mask_3d, output, torch.zeros_like(output))


    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
    ):
        bs, num_pred, num_cls = cls_pred.shape

        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)

        instance_reg_weights = []
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
                assign = linear_sum_assignment(cost)
                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                indices.append([None, None])

        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue
            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]
        return output_cls_target, output_box_target, output_reg_weights

    def _compute_cls_diff(self, cls_pred):
        p = cls_pred.sigmoid()
        neg_cost = -(1 - p + self.eps).log() * (1 - self.alpha) * p.pow(self.gamma)
        pos_cost = -(p + self.eps).log() * self.alpha * (1 - p).pow(self.gamma)
        return (pos_cost - neg_cost) * self.cls_weight  # [bs, num_pred, num_cls]

    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        diff = self._compute_cls_diff(cls_pred)
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                cost.append(diff[i].index_select(1, cls_target[i]))
            else:
                cost.append(None)
        return cost

    def _box_cost_flat(self, pred_per_gt, flat_target, flat_weights, reg_weights):
        diff = torch.abs(pred_per_gt - flat_target[:, None, :])
        weighted = diff * flat_weights[:, None, :] * reg_weights
        return weighted.sum(dim=-1) * self.box_weight  # [total_gt, num_pred]

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        lengths = [t.shape[0] for t in box_target]
        total_gt = sum(lengths)
        if total_gt == 0:
            return [None] * bs

        flat_target = torch.cat(box_target, dim=0)
        flat_weights = torch.cat(instance_reg_weights, dim=0)
        batch_idx = torch.repeat_interleave(
            torch.arange(bs, device=box_pred.device),
            torch.tensor(lengths, device=box_pred.device),
        )
        pred_per_gt = box_pred.index_select(0, batch_idx)
        reg_weights = box_pred.new_tensor(self.reg_weights)

        cost_flat = self._box_cost_flat(
            pred_per_gt, flat_target, flat_weights, reg_weights
        )

        parts = cost_flat.split(lengths, dim=0)
        return [
            p.t().contiguous() if L > 0 else None
            for p, L in zip(parts, lengths)
        ]


    def _assign_and_scatter(
        self,
        box_cost,
        dn_anchor,
        box_target,
        cls_target,
        gt_instance_id,
    ):
        assignment = _batch_linear_assignment_op(box_cost)  # [G*bs, W]
        valid = assignment >= 0                          # [G*bs, W]
        safe_gt = assignment.clamp(min=0)                # [G*bs, W]

        state_dims = box_target.size(-1)
        gathered_box = box_target.gather(
            1, safe_gt.unsqueeze(-1).expand(-1, -1, state_dims)
        )                                                # [G*bs, W, D]
        dn_box_target = torch.where(
            valid.unsqueeze(-1),
            gathered_box,
            torch.zeros_like(gathered_box),
        )

        gathered_cls = cls_target.gather(1, safe_gt)     # [G*bs, W]
        dn_cls_target = torch.where(
            valid,
            gathered_cls,
            torch.full_like(gathered_cls, -3),
        )

        if gt_instance_id is not None:
            gathered_id = gt_instance_id.gather(1, safe_gt)
            dn_id_target = torch.where(
                valid,
                gathered_id,
                torch.full_like(gathered_id, -1),
            )
        else:
            dn_id_target = None

        num_dn_groups = self.num_dn_groups
        bs = box_target.size(0) // num_dn_groups

        def group_to_batch(x):
            return (
                x.unflatten(0, (num_dn_groups, bs))
                .transpose(0, 1)
                .flatten(1, 2)
            )

        dn_anchor = group_to_batch(dn_anchor)
        dn_box_target = group_to_batch(dn_box_target)
        dn_cls_target = group_to_batch(dn_cls_target)
        if dn_id_target is not None:
            dn_id_target = group_to_batch(dn_id_target)
        neg_valid_target = group_to_batch(
            torch.cat([cls_target, cls_target], dim=1)
        )

        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            dn_id_target,
            neg_valid_target,
        )

    def _box_cost_fix(self, box_pred, box_target, instance_reg_weights):

        reg_weights = box_pred.new_tensor(self.reg_weights)
        weighted_abs_error = (
            torch.abs(box_pred[:, :, None, :] - box_target[:, None, :, :])
            * instance_reg_weights[:, None, :, :]
            * reg_weights
        )
        return weighted_abs_error.sum(dim=-1) * self.box_weight


    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        has_gt = any(x.shape[0] > 0 for x in cls_target)
        if not has_gt:
            return None
        if self.max_dn_gt > 0:
            max_dn_gt = self.max_dn_gt
            cls_target = [x[:max_dn_gt] for x in cls_target]
            box_target = [x[:max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[:max_dn_gt]for x in gt_instance_id]
        else:
            max_dn_gt = max(x.shape[0] for x in cls_target)
        cls_target = torch.stack([F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                                  for x in cls_target], dim=0)
        box_target = torch.stack([F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]), value=0)
                                  for x in box_target], dim=0
                                ).to(device=cls_target.device,non_blocking=True)
        valid_mask = cls_target >= 0

        box_target = self.encode_reg_target_fix(
            box_target,
            valid_mask,
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                 for x in gt_instance_id], dim=0
                 ).to(device=cls_target.device)

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1 # 与box_target相同shape
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost_fix(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            dn_id_target,
            neg_valid_target,
        ) = self._assign_and_scatter(
            box_cost,
            dn_anchor,
            box_target,
            cls_target,
            gt_instance_id,
        )
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            valid_mask |= (neg_valid_target >= 0) & (dn_cls_target == -3)
        group_idx = torch.arange(
            self.num_dn_groups,
            device=dn_anchor.device,
        ).repeat_interleave(num_gt)

        attn_mask = group_idx[:, None] != group_idx[None, :]


        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            -torch.ones_like(self.dn_metas["dn_cls_target"]),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        group_order = torch.randperm(num_dn_groups).argsort()
        # Take the same groups in the same order as the boolean mask.
        group_indices = group_order[:self.num_temp_dn_groups].sort().values
        group_indices = group_indices.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        ).index_select(1, group_indices)
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        ).index_select(1, group_indices)
        dn_cls_target = dn_cls_target.reshape(
            bs, num_dn_groups, num_temp_dn
        ).index_select(1, group_indices)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn).index_select(
            1, group_indices
        )
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            ).index_select(1, group_indices)
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )
