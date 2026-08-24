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

"""LightOp replacements for MMCV's low-level MSDA entry points."""

from __future__ import annotations

from lightop import op as _lightop


def _require_callable(owner, name: str, owner_name: str):
    value = getattr(owner, name, None)
    if not callable(value):
        raise RuntimeError(f"required callable is unavailable: {owner_name}.{name}")
    return value


_msda_forward = _require_callable(
    _lightop, "ms_deform_attn_forward", "lightop.op"
)
_msda_backward = _require_callable(
    _lightop, "ms_deform_attn_backward", "lightop.op"
)


def ms_deform_attn_forward(
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
):
    """Match ``mmcv._ext.ms_deform_attn_forward`` using LightOp."""

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
    """Match MMCV's in-place MSDA backward contract using LightOp."""

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
