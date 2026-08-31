# Derived from Sparse4D: projects/mmdet3d_plugin/ops/deformable_aggregation.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

import torch
from torch.autograd.function import Function, once_differentiable

from turbo_physai.ops import deformable_aggregation_forward, deformable_aggregation_backward


class DeformableAggregationFunction(Function):
    @staticmethod
    def forward(
        ctx,
        mc_ms_feat,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    ):
        # output: [bs, num_pts, num_embeds]
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_forward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        ctx.save_for_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
        ) = ctx.saved_tensors
        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()

        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_backward(
            mc_ms_feat,
            spatial_shape,
            scale_start_index,
            sampling_location,
            weights,
            grad_output.contiguous(),
            grad_mc_ms_feat,
            grad_sampling_location,
            grad_weights,
        )
        return (
            grad_mc_ms_feat,
            None,
            None,
            grad_sampling_location,
            grad_weights,
        )

def deformable_aggregation_function(
    feature_maps,
    spatial_shape,
    scale_start_index,
    sampling_location,
    weights,
):
    return DeformableAggregationFunction.apply(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sampling_location,
        weights,
    )
