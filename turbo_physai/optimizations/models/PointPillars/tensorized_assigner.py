# Portions are adapted from the PointPillars HCU optimization project:
# Source: PointPillars/mmdetection3d/mmdet3d/models/task_modules/assigners/tensorized_max_3d_iou_assigner.py
# Revision: bd37518d0408bcea4ac879a90256ffb8b8628258
# Project license: Apache-2.0
# Project copyright: Copyright 2018-2019 Open-MMLab. All rights reserved.
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
# SPDX-License-Identifier: Apache-2.0

"""Tensorized MaxIoU assignment replacement for PointPillars."""

from functools import wraps
import os

import torch
from mmdet.models.task_modules import AssignResult


def _tensorized_assign_impl(overlaps, gt_labels, pos_iou_thr, neg_iou_thr,
                            min_pos_iou):
    """Return assignment tensors using vectorized low-quality matching."""
    max_overlaps, argmax_overlaps = overlaps.max(dim=0)
    gt_max_overlaps, _ = overlaps.max(dim=1)

    assigned_gt_inds = torch.full_like(max_overlaps, -1, dtype=torch.long)
    negative = (max_overlaps >= 0) & (max_overlaps < neg_iou_thr)
    assigned_gt_inds = torch.where(
        negative, torch.zeros_like(assigned_gt_inds), assigned_gt_inds
    )
    positive = max_overlaps >= pos_iou_thr
    assigned_gt_inds = torch.where(
        positive, argmax_overlaps + 1, assigned_gt_inds
    )

    valid_gt = gt_max_overlaps >= min_pos_iou
    matches = (overlaps == gt_max_overlaps[:, None]) & valid_gt[:, None]
    gt_ids = torch.arange(
        1, overlaps.shape[0] + 1, device=overlaps.device, dtype=torch.int32
    )[:, None]
    winners = torch.where(
        matches, gt_ids, torch.zeros_like(gt_ids)
    ).amax(dim=0).long()
    assigned_gt_inds = torch.where(
        winners > 0, winners, assigned_gt_inds
    )

    assigned_labels = None
    if gt_labels is not None:
        safe_index = (assigned_gt_inds - 1).clamp_min(0)
        assigned_labels = torch.where(
            assigned_gt_inds > 0,
            gt_labels[safe_index],
            gt_labels.new_full(assigned_gt_inds.shape, -1),
        )
    return assigned_gt_inds, max_overlaps, assigned_labels


def tensorized_assign_wrt_overlaps_wrapper(original, options):
    """Replace only the supported scalar-threshold PointPillars path."""
    options = options or {}
    enable_compile = options.get("compile", True)
    compiled = None

    @wraps(original)
    def wrapped(self, overlaps, gt_labels):
        nonlocal compiled
        supported = (
            isinstance(self.neg_iou_thr, (float, int))
            and self.match_low_quality
            and self.gt_max_assign_all
        )
        if (not supported or overlaps.shape[0] == 0
                or overlaps.shape[1] == 0):
            return original(self, overlaps, gt_labels)

        impl = _tensorized_assign_impl
        if (enable_compile and overlaps.device.type == "cuda"
                and os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") != "1"):
            if compiled is None:
                compiled = torch.compile(impl, backend="inductor", dynamic=True,
                                         fullgraph=True)
            impl = compiled
        gt_inds, max_overlaps, labels = impl(
            overlaps,
            gt_labels,
            self.pos_iou_thr,
            self.neg_iou_thr,
            self.min_pos_iou,
        )
        return AssignResult(
            num_gts=overlaps.shape[0],
            gt_inds=gt_inds,
            max_overlaps=max_overlaps,
            labels=labels,
        )

    return wrapped
