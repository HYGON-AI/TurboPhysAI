# Portions are adapted from the PointPillars HCU optimization project:
# Source: PointPillars/mmdetection3d/mmdet3d/models/dense_heads/unique_free_anchor3d_head.py
# Revision: bd37518d0408bcea4ac879a90256ffb8b8628258
# Project license: Apache-2.0
# Project copyright: Copyright 2018-2019 Open-MMLab. All rights reserved.
#
# Portions are derived from OpenMMLab MMDetection3D:
# https://github.com/open-mmlab/mmdetection3d
# File: mmdet3d/models/task_modules/samplers/pseudosample.py
# Commit: fe25f7a51d36e3702f961e198894580d83c4387b
# Copyright (c) OpenMMLab. All rights reserved.
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
# SPDX-License-Identifier: Apache-2.0

"""PseudoSampler replacement that omits redundant unique operations."""

from functools import wraps

import torch
from mmdet3d.models.task_modules.samplers import SamplingResult


def unique_free_sampler_wrapper(original, options):
    """Preserve upstream SamplingResult behavior without sorting indices."""

    @wraps(original)
    def wrapped(self, assign_result, pred_instances, gt_instances,
                *args, **kwargs):
        gt_bboxes = gt_instances.bboxes_3d
        priors = pred_instances.priors
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False
        ).squeeze(-1)
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False
        ).squeeze(-1)
        gt_flags = priors.new_zeros(priors.shape[0], dtype=torch.uint8)
        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False,
        )

    return wrapped
