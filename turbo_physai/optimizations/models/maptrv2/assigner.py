# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py
# Upstream: projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) 2022 Hust Vision Lab
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Complete static Hungarian assignment transfer for MapTRv2.

The reference implementation changes the GT contract between
``MapTRv2Head.loss``, ``_get_target_single`` and ``MapTRAssigner.assign``.
A single symbol cannot be replaced safely, so this module supplies the three
changed existing methods as one atomic Group.
"""

from __future__ import annotations

import os


STATIC_GT_COUNT = 200


def pad_to_static_list(tensors, pad_value=0, device=None):
    import torch

    results = []
    for tensor in tensors:
        output = torch.full(
            (STATIC_GT_COUNT,) + tensor.shape[1:],
            pad_value,
            device=device,
            dtype=tensor.dtype,
        )
        mask = torch.zeros(
            STATIC_GT_COUNT, dtype=torch.bool, device=device
        )
        length = tensor.size(0)
        if length > STATIC_GT_COUNT:
            raise ValueError(
                f"GT count {length} exceeds static capacity "
                f"{STATIC_GT_COUNT}"
            )
        output[:length] = tensor
        mask[:length] = True
        results.append((output, mask, length))
    return results


def get_label_result(
    self,
    sampling_result,
    gt_bboxes,
    gt_labels,
    bbox_pred,
    order_index,
    pts_pred,
    gt_shifts_pts,
):
    import torch

    num_bboxes = bbox_pred.size(0)
    gt_channels = gt_bboxes.shape[-1]
    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    assigned_gt_inds = sampling_result.pos_assigned_gt_inds
    labels = gt_bboxes.new_full(
        (num_bboxes,), self.num_classes, dtype=torch.long
    )
    labels[pos_inds] = gt_labels[assigned_gt_inds]
    label_weights = gt_bboxes.new_ones(num_bboxes)
    bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_channels]
    bbox_weights = torch.zeros_like(bbox_pred)
    bbox_weights[pos_inds] = 1.0
    if order_index is None:
        assigned_shift = gt_labels[assigned_gt_inds]
    else:
        assigned_shift = order_index[pos_inds, assigned_gt_inds]
    pts_targets = pts_pred.new_zeros(
        (pts_pred.size(0), pts_pred.size(1), pts_pred.size(2))
    )
    pts_weights = torch.zeros_like(pts_targets)
    pts_weights[pos_inds] = 1.0
    bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
    pts_targets[pos_inds] = gt_shifts_pts[
        assigned_gt_inds, assigned_shift, :, :
    ]
    return (
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        pts_targets,
        pts_weights,
        pos_inds,
        neg_inds,
    )


def get_target_single(
    self,
    cls_score,
    bbox_pred,
    pts_pred,
    gt_labels,
    gt_bboxes,
    gt_shifts_pts,
    gt_bboxes_ignore=None,
):
    assign_result, order_index = self.assigner.assign(
        bbox_pred,
        cls_score,
        pts_pred,
        gt_bboxes,
        gt_labels,
        gt_shifts_pts,
        gt_bboxes_ignore,
    )
    sampling_result = self.sampler.sample(
        assign_result, bbox_pred, gt_bboxes[0]
    )
    return get_label_result(
        self,
        sampling_result,
        gt_bboxes[0],
        gt_labels[0],
        bbox_pred,
        order_index,
        pts_pred,
        gt_shifts_pts[0],
    )


def _hungarian_match_impl(
    cost, gt_labels, assigned_gt_inds, assigned_labels, num_gts, device
):
    import torch
    from mmdet.core.bbox.assigners import AssignResult

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            'Please run "pip install scipy" to install scipy first.'
        ) from exc
    row_indices, column_indices = linear_sum_assignment(cost.detach().cpu())
    matched_rows = torch.as_tensor(row_indices, device=device)
    matched_columns = torch.as_tensor(column_indices, device=device)
    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_rows] = matched_columns + 1
    assigned_labels[matched_rows] = gt_labels[0][matched_columns]
    return AssignResult(
        num_gts, assigned_gt_inds, None, labels=assigned_labels
    )


_DISABLED_HUNGARIAN = None
_COMPILED_ASSIGN = None
_STATIC_LOSS = None


def _get_hungarian_match():
    global _DISABLED_HUNGARIAN
    if _DISABLED_HUNGARIAN is None:
        import torch

        _DISABLED_HUNGARIAN = torch._dynamo.disable(_hungarian_match_impl)
    return _DISABLED_HUNGARIAN


def _assign_impl(
    self,
    bbox_pred,
    cls_pred,
    pts_pred,
    gt_bboxes,
    gt_labels,
    gt_pts,
    gt_bboxes_ignore=None,
    eps=1e-7,
):
    del eps
    import torch
    import torch.nn.functional as functional
    from projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head import (
        denormalize_2d_bbox,
        normalize_2d_bbox,
        normalize_2d_pts,
        normalize_3d_pts,
    )

    if gt_bboxes_ignore is not None:
        raise AssertionError(
            "Only case when gt_bboxes_ignore is None is supported."
        )
    if bbox_pred.shape[-1] != 4:
        raise AssertionError("Only support bbox pred shape is 4 dims")
    num_bboxes = bbox_pred.size(0)
    assigned_gt_inds = bbox_pred.new_full(
        (num_bboxes,), -1, dtype=torch.long
    )
    assigned_labels = bbox_pred.new_full(
        (num_bboxes,), -1, dtype=torch.long
    )
    cls_cost = self.cls_cost(cls_pred, gt_labels[0])
    normalized_gt_bboxes = normalize_2d_bbox(gt_bboxes[0], self.pc_range)
    _, num_orders, num_pts_per_gtline, _ = gt_pts[0].shape
    normalizer = (
        normalize_3d_pts
        if self.z_cfg["gt_z_flag"]
        else normalize_2d_pts
    )
    normalized_gt_pts = normalizer(gt_pts[0], self.pc_range)
    if pts_pred.size(1) != num_pts_per_gtline:
        pts_pred_interpolated = functional.interpolate(
            pts_pred.permute(0, 2, 1),
            size=(num_pts_per_gtline,),
            mode="linear",
            align_corners=True,
        )
        pts_pred_interpolated = pts_pred_interpolated.permute(
            0, 2, 1
        ).contiguous()
    else:
        pts_pred_interpolated = pts_pred
    bboxes = denormalize_2d_bbox(bbox_pred, self.pc_range)
    pts_cost_ordered = self.pts_cost(
        pts_pred_interpolated, normalized_gt_pts
    )
    pts_cost_ordered = pts_cost_ordered.view(
        num_bboxes, gt_bboxes[0].size(0), num_orders
    )
    pts_cost, order_index = torch.min(pts_cost_ordered, 2)
    reg_cost = self.reg_cost(
        bbox_pred[:, :4], normalized_gt_bboxes[:, :4]
    )
    iou_cost = self.iou_cost(bboxes, gt_bboxes[0])
    cost = cls_cost + reg_cost + iou_cost + pts_cost
    assign_result = _get_hungarian_match()(
        cost[:, gt_bboxes[1]],
        gt_labels,
        assigned_gt_inds,
        assigned_labels,
        gt_bboxes[2],
        bbox_pred.device,
    )
    return assign_result, order_index


def _assign_no_grad(
    self,
    bbox_pred,
    cls_pred,
    pts_pred,
    gt_bboxes,
    gt_labels,
    gt_pts,
    gt_bboxes_ignore=None,
    eps=1e-7,
):
    import torch

    with torch.no_grad():
        return _assign_impl(
            self,
            bbox_pred,
            cls_pred,
            pts_pred,
            gt_bboxes,
            gt_labels,
            gt_pts,
            gt_bboxes_ignore,
            eps,
        )


def assign(
    self,
    bbox_pred,
    cls_pred,
    pts_pred,
    gt_bboxes,
    gt_labels,
    gt_pts,
    gt_bboxes_ignore=None,
    eps=1e-7,
):
    global _COMPILED_ASSIGN
    if os.getenv("TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE", "1") != "1":
        return _assign_no_grad(
            self,
            bbox_pred,
            cls_pred,
            pts_pred,
            gt_bboxes,
            gt_labels,
            gt_pts,
            gt_bboxes_ignore,
            eps,
        )
    if _COMPILED_ASSIGN is None:
        import torch

        _COMPILED_ASSIGN = torch.compile(
            _assign_no_grad,
            mode="max-autotune-no-cudagraphs",
        )
    return _COMPILED_ASSIGN(
        self,
        bbox_pred,
        cls_pred,
        pts_pred,
        gt_bboxes,
        gt_labels,
        gt_pts,
        gt_bboxes_ignore,
        eps,
    )


def _shift_points(self, gt_vecs):
    attribute = {
        "v0": "shift_fixed_num_sampled_points",
        "v1": "shift_fixed_num_sampled_points_v1",
        "v2": "shift_fixed_num_sampled_points_v2",
        "v3": "shift_fixed_num_sampled_points_v3",
        "v4": "shift_fixed_num_sampled_points_v4",
    }.get(self.gt_shift_pts_pattern)
    if attribute is None:
        raise NotImplementedError
    return [getattr(item, attribute) for item in gt_vecs]


def _static_loss_impl(
    self,
    gt_bboxes_list,
    gt_labels_list,
    gt_seg_mask,
    gt_pv_seg_mask,
    preds_dicts,
    gt_bboxes_ignore=None,
    img_metas=None,
):
    del img_metas
    import torch
    from mmdet.core import multi_apply

    if gt_bboxes_ignore is not None:
        raise AssertionError(
            f"{self.__class__.__name__} only supports "
            "gt_bboxes_ignore setting to None."
        )
    gt_vecs_list = gt_bboxes_list
    all_cls_scores = preds_dicts["all_cls_scores"]
    all_bbox_preds = preds_dicts["all_bbox_preds"]
    all_pts_preds = preds_dicts["all_pts_preds"]
    enc_cls_scores = preds_dicts["enc_cls_scores"]
    enc_bbox_preds = preds_dicts["enc_bbox_preds"]
    enc_pts_preds = preds_dicts["enc_pts_preds"]
    num_dec_layers = len(all_cls_scores)
    device = gt_labels_list[0].device
    gt_bboxes_list = [
        item.bbox.to(device) for item in gt_vecs_list
    ]
    gt_shifts_pts_list = [
        item.to(device) for item in _shift_points(self, gt_vecs_list)
    ]
    static_bboxes = pad_to_static_list(gt_bboxes_list, device=device)
    static_labels = pad_to_static_list(gt_labels_list, device=device)
    static_points = pad_to_static_list(gt_shifts_pts_list, device=device)
    all_gt_bboxes = [static_bboxes] * num_dec_layers
    all_gt_labels = [static_labels] * num_dec_layers
    all_gt_points = [static_points] * num_dec_layers
    all_ignored = [gt_bboxes_ignore] * num_dec_layers
    losses = multi_apply(
        self.loss_single,
        all_cls_scores,
        all_bbox_preds,
        all_pts_preds,
        all_gt_bboxes,
        all_gt_labels,
        all_gt_points,
        all_ignored,
    )
    losses_cls, losses_bbox, losses_iou, losses_pts, losses_dir = losses
    loss_dict = {}
    if self.aux_seg["use_aux_seg"]:
        if self.aux_seg["bev_seg"] and preds_dicts["seg"] is not None:
            num_images = preds_dicts["seg"].size(0)
            targets = torch.stack(
                [gt_seg_mask[index] for index in range(num_images)]
            )
            loss_dict["loss_seg"] = self.loss_seg(
                preds_dicts["seg"], targets.float()
            )
        if self.aux_seg["pv_seg"] and preds_dicts["pv_seg"] is not None:
            num_images = preds_dicts["pv_seg"].size(0)
            targets = torch.stack(
                [gt_pv_seg_mask[index] for index in range(num_images)]
            )
            loss_dict["loss_pv_seg"] = self.loss_pv_seg(
                preds_dicts["pv_seg"], targets.float()
            )
    if enc_cls_scores is not None:
        binary_labels = [
            (torch.zeros_like(item[0]), item[1], item[2])
            for item in static_labels
        ]
        encoded_losses = self.loss_single(
            enc_cls_scores,
            enc_bbox_preds,
            enc_pts_preds,
            static_bboxes,
            binary_labels,
            static_points,
            gt_bboxes_ignore,
        )
        (
            loss_dict["enc_loss_cls"],
            loss_dict["enc_loss_bbox"],
            loss_dict["enc_loss_iou"],
            loss_dict["enc_loss_pts"],
            loss_dict["enc_loss_dir"],
        ) = encoded_losses
    loss_dict["loss_cls"] = losses_cls[-1]
    loss_dict["loss_bbox"] = losses_bbox[-1]
    loss_dict["loss_iou"] = losses_iou[-1]
    loss_dict["loss_pts"] = losses_pts[-1]
    loss_dict["loss_dir"] = losses_dir[-1]
    for index, values in enumerate(
        zip(
            losses_cls[:-1],
            losses_bbox[:-1],
            losses_iou[:-1],
            losses_pts[:-1],
            losses_dir[:-1],
        )
    ):
        (
            loss_dict[f"d{index}.loss_cls"],
            loss_dict[f"d{index}.loss_bbox"],
            loss_dict[f"d{index}.loss_iou"],
            loss_dict[f"d{index}.loss_pts"],
            loss_dict[f"d{index}.loss_dir"],
        ) = values
    return loss_dict


def _get_static_loss():
    global _STATIC_LOSS
    if _STATIC_LOSS is None:
        from mmcv.runner import force_fp32

        _STATIC_LOSS = force_fp32(apply_to=("preds_dicts",))(
            _static_loss_impl
        )
    return _STATIC_LOSS


def static_loss(
    self,
    gt_bboxes_list,
    gt_labels_list,
    gt_seg_mask,
    gt_pv_seg_mask,
    preds_dicts,
    gt_bboxes_ignore=None,
    img_metas=None,
):
    return _get_static_loss()(
        self,
        gt_bboxes_list,
        gt_labels_list,
        gt_seg_mask,
        gt_pv_seg_mask,
        preds_dicts,
        gt_bboxes_ignore,
        img_metas,
    )


def hungarian_match(
    cost, gt_labels, assigned_gt_inds, assigned_labels, num_gts, device
):
    """Compile-time-stable entry that stays outside Dynamo graphs."""

    return _get_hungarian_match()(
        cost, gt_labels, assigned_gt_inds, assigned_labels, num_gts, device
    )


__all__ = [
    "assign",
    "get_target_single",
    "hungarian_match",
    "pad_to_static_list",
    "static_loss",
]
