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

"""Reusable LightOp implementation of multi-scale deformable attention.

The module stays importable when ``lightop`` is missing so that the caller can
decide what to do: ``mmcv.msda`` uses :func:`lightop_available` as its
``runtime_condition`` and falls back to the mmcv extension, which mirrors the
reference implementation's ``try: from lightop import op`` / ``except
ImportError`` pair.
"""

from __future__ import annotations


_REQUIRED_FUNCTIONS = ("ms_deform_attn_forward", "ms_deform_attn_backward")
_LIGHTOP_UNSET = object()
_lightop = _LIGHTOP_UNSET


def _load_lightop():
    global _lightop
    if _lightop is _LIGHTOP_UNSET:
        try:
            from lightop import op
        except ImportError:  # pragma: no cover - depends on the installed image
            _lightop = None
        else:
            _lightop = op
    return _lightop


def lightop_available(*_args, **_kwargs) -> bool:
    """Return whether the LightOp MSDA backend can be used."""

    lightop = _load_lightop()
    return all(
        callable(getattr(lightop, name, None)) for name in _REQUIRED_FUNCTIONS
    )


def _require_lightop():
    lightop = _load_lightop()
    if not all(
        callable(getattr(lightop, name, None)) for name in _REQUIRED_FUNCTIONS
    ):
        raise RuntimeError(
            "lightop is not installed; the mmcv.msda Group falls back to the "
            "mmcv MSDeformAttn extension and must not reach this operator"
        )
    return lightop


def ms_deform_attn_forward(
    value,
    value_spatial_shapes,
    value_level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
):
    """Execute the LightOp MSDA forward operator."""

    return _require_lightop().ms_deform_attn_forward(
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
    grad_sampling_loc,
    grad_attn_weight,
    im2col_step,
):
    """Execute the LightOp MSDA backward operator."""

    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    _require_lightop().ms_deform_attn_backward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        grad_output,
        grad_value,
        grad_sampling_loc,
        grad_attn_weight,
        im2col_step,
    )


__all__ = [
    "lightop_available",
    "ms_deform_attn_backward",
    "ms_deform_attn_forward",
]
