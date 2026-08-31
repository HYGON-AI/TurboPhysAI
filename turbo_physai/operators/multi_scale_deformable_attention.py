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

"""Reusable LightOp implementation of multi-scale deformable attention."""

from __future__ import annotations

from lightop import op as _lightop


_msda_forward = _lightop.ms_deform_attn_forward
_msda_backward = _lightop.ms_deform_attn_backward


def ms_deform_attn_forward(
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
):
    """Execute the LightOp MSDA forward operator."""

    return _msda_forward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    )


def ms_deform_attn_backward(
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
):
    """Execute the LightOp MSDA backward operator."""

    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
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


__all__ = ["ms_deform_attn_forward", "ms_deform_attn_backward"]
