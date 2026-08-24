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

"""Compile-friendly LightOp replacement for BEVFormer's local MSDA classes."""

from __future__ import annotations

from typing import Tuple

import torch
from lightop import op as _lightop


def _require_callable(owner, name: str, owner_name: str):
    value = getattr(owner, name, None)
    if not callable(value):
        raise RuntimeError(f"required callable is unavailable: {owner_name}.{name}")
    return value


_require_callable(torch.library, "custom_op", "torch.library")
_require_callable(torch.library, "register_autograd", "torch.library")
_msda_forward = _require_callable(
    _lightop, "ms_deform_attn_forward", "lightop.op"
)
_msda_backward = _require_callable(
    _lightop, "ms_deform_attn_backward", "lightop.op"
)


@torch.library.custom_op("mmcv::ms_deform_attn", mutates_args=())
def ms_deform_attn(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    return _msda_forward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )


@ms_deform_attn.register_fake
def _ms_deform_attn_fake(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int,
) -> torch.Tensor:
    del value_spatial_shapes, value_level_start_index, attention_weights, im2col_step
    batch_size = value.shape[0]
    num_queries = sampling_locations.shape[1]
    embed_dims = value.shape[3] * sampling_locations.shape[2]
    return torch.empty(
        batch_size,
        num_queries,
        embed_dims,
        device=value.device,
        dtype=value.dtype,
    )


@torch.library.custom_op("mmcv::ms_deform_attn_backward", mutates_args=())
def ms_deform_attn_backward(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    grad_value = torch.zeros_like(value)
    grad_sampling_locations = torch.zeros_like(sampling_locations)
    grad_attention_weights = torch.zeros_like(attention_weights)
    _msda_backward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        grad_output,
        grad_value,
        grad_sampling_locations,
        grad_attention_weights,
        im2col_step,
    )
    return grad_value, grad_sampling_locations, grad_attention_weights


@ms_deform_attn_backward.register_fake
def _ms_deform_attn_backward_fake(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    value_level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    grad_output: torch.Tensor,
    im2col_step: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del value_spatial_shapes, value_level_start_index, grad_output, im2col_step
    return (
        torch.empty_like(value),
        torch.empty_like(sampling_locations),
        torch.empty_like(attention_weights),
    )


def _setup_context(ctx, inputs, output) -> None:
    del output
    (
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ) = inputs
    ctx.save_for_backward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
    )
    ctx.im2col_step = im2col_step


def _backward(ctx, grad_output):
    (
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
    ) = ctx.saved_tensors
    grad_value, grad_sampling_locations, grad_attention_weights = (
        ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
    )
    return (
        grad_value,
        None,
        None,
        grad_sampling_locations,
        grad_attention_weights,
        None,
    )


torch.library.register_autograd(
    "mmcv::ms_deform_attn", _backward, setup_context=_setup_context
)


class MultiScaleDeformableAttnFunction_fp16:
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def apply(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        if torch.is_autocast_enabled():
            value = value.to(torch.float16)
            sampling_locations = sampling_locations.to(torch.float16)
            attention_weights = attention_weights.to(torch.float16)
        return ms_deform_attn(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )


class MultiScaleDeformableAttnFunction_fp32:
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def apply(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        if torch.is_autocast_enabled():
            value = value.to(torch.float32)
            sampling_locations = sampling_locations.to(torch.float32)
            attention_weights = attention_weights.to(torch.float32)
        return ms_deform_attn(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step,
        )


__all__ = [
    "MultiScaleDeformableAttnFunction_fp16",
    "MultiScaleDeformableAttnFunction_fp32",
    "ms_deform_attn",
    "ms_deform_attn_backward",
]
