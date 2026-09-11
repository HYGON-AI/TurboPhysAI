# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime input conditions for the LightOp MMCV MSDA adapter."""

from __future__ import annotations


def is_supported_msda_forward(
    value, value_spatial_shapes, value_level_start_index,
    sampling_locations, attention_weights, im2col_step,
):
    """Check tensor metadata without copying or synchronizing device data."""
    import torch

    tensors = (value, value_spatial_shapes, value_level_start_index,
               sampling_locations, attention_weights)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if not torch.version.hip or not value.is_cuda:
        return False
    if not all(t.device == value.device and t.is_contiguous() for t in tensors):
        return False
    if value.dtype not in (torch.float32, torch.float64):
        return False
    if sampling_locations.dtype != value.dtype or attention_weights.dtype != value.dtype:
        return False
    if value_spatial_shapes.dtype != torch.int64 or value_level_start_index.dtype != torch.int64:
        return False
    if tuple(t.dim() for t in tensors) != (4, 2, 1, 6, 5):
        return False
    batch, _, heads, _ = value.shape
    if any(size <= 0 for tensor in tensors for size in tensor.shape):
        return False
    if not isinstance(im2col_step, int) or isinstance(im2col_step, bool) or im2col_step <= 0:
        return False
    if batch % min(batch, im2col_step):
        return False
    levels = value_spatial_shapes.shape[0]
    return (
        value_spatial_shapes.shape[1] == 2
        and value_level_start_index.shape == (levels,)
        and sampling_locations.shape[0] == batch
        and sampling_locations.shape[2:4] == (heads, levels)
        and sampling_locations.shape[-1] == 2
        and attention_weights.shape == sampling_locations.shape[:-1]
    )


def is_supported_msda_backward(
    value, value_spatial_shapes, value_level_start_index,
    sampling_locations, attention_weights, grad_output,
    grad_value, grad_sampling_loc, grad_attn_weight, im2col_step,
):
    """Use the forward contract and check backward gradient buffers."""
    import torch

    if not is_supported_msda_forward(
        value, value_spatial_shapes, value_level_start_index,
        sampling_locations, attention_weights, im2col_step,
    ):
        return False
    gradients = (grad_output, grad_value, grad_sampling_loc, grad_attn_weight)
    if not all(isinstance(tensor, torch.Tensor) for tensor in gradients):
        return False
    if not all(t.device == value.device and t.dtype == value.dtype for t in gradients):
        return False
    # The adapter makes grad_output contiguous; output buffers must already be so.
    if not all(t.is_contiguous() for t in gradients[1:]):
        return False
    return (
        grad_output.shape == (value.shape[0], sampling_locations.shape[1],
                              value.shape[2] * value.shape[3])
        and grad_value.shape == value.shape
        and grad_sampling_loc.shape == sampling_locations.shape
        and grad_attn_weight.shape == attention_weights.shape
    )
