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
