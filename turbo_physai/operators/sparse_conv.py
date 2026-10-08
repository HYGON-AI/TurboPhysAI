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

"""Sparse-convolution compatibility operators backed by the bundled extension."""

from __future__ import annotations

import torch


def _ops():
    from turbo_physai import _C

    return _C


def _conv_output_size(input_size, kernel_size, stride, padding, dilation):
    output = []
    for size, kernel, step, pad, dilate in zip(
        input_size, kernel_size, stride, padding, dilation
    ):
        if kernel == -1:
            output.append(1)
        else:
            output.append(
                (size + 2 * pad - dilate * (kernel - 1) - 1) // step + 1
            )
    return output


def _deconv_output_size(
    input_size, kernel_size, stride, padding, dilation, output_padding
):
    output = []
    for size, kernel, step, pad, dilate, out_pad in zip(
        input_size,
        kernel_size,
        stride,
        padding,
        dilation,
        output_padding,
    ):
        if kernel == -1:
            raise ValueError("deconvolution does not support kernel_size < 0")
        output.append(
            (size - 1) * step - 2 * pad + kernel + out_pad
        )
    return output


def _check_dtype(features):
    if features.dtype == torch.float32:
        return "fp32"
    if features.dtype == torch.float16:
        return "half"
    raise TypeError(f"sparse operators do not support {features.dtype}")


def get_indice_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize=3,
    stride=1,
    padding=0,
    dilation=1,
    out_padding=0,
    subm=False,
    transpose=False,
    grid=None,
):
    """Call the bundled extension that canonicalizes generated indice pairs."""

    indices = indices.contiguous()
    if isinstance(grid, torch.Tensor) and not grid.is_contiguous():
        raise ValueError("grid must be contiguous because the native operator mutates it")

    ndim = indices.shape[1] - 1

    def dimensions(value):
        return list(value) if isinstance(value, (list, tuple)) else [value] * ndim

    ksize = dimensions(ksize)
    stride = dimensions(stride)
    padding = dimensions(padding)
    dilation = dimensions(dilation)
    out_padding = dimensions(out_padding)
    for dilate, step in zip(dilation, stride):
        if step != 1 and dilate != 1:
            raise ValueError("stride and dilation cannot both exceed one")

    if subm:
        output_shape = spatial_shape
    elif transpose:
        output_shape = _deconv_output_size(
            spatial_shape, ksize, stride, padding, dilation, out_padding
        )
    else:
        output_shape = _conv_output_size(
            spatial_shape, ksize, stride, padding, dilation
        )

    extension = _ops()
    if grid is None:
        function = getattr(extension, f"get_indice_pairs_{ndim}d", None)
        if function is None:
            raise NotImplementedError(f"unsupported sparse convolution rank: {ndim}")
        return function(
            indices,
            batch_size,
            output_shape,
            spatial_shape,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            int(subm),
            int(transpose),
        )

    function = getattr(extension, f"get_indice_pairs_grid_{ndim}d", None)
    if function is None:
        raise NotImplementedError(
            f"unsupported sparse convolution grid rank: {ndim}"
        )
    return function(
        indices,
        grid,
        batch_size,
        output_shape,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        int(subm),
        int(transpose),
    )


class _IndiceConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features,
        filters,
        indice_pairs,
        indice_num,
        num_act_out,
        inverse=False,
        subm=False,
    ):
        suffix = _check_dtype(features)
        if filters.dtype != features.dtype:
            raise TypeError("features and filters must have the same dtype")
        features = features.contiguous()
        filters = filters.contiguous()
        indice_pairs = indice_pairs.contiguous()
        indice_num = indice_num.contiguous()
        extension = _ops()
        function = getattr(extension, f"indice_conv_{suffix}")
        output = function(
            features,
            filters,
            indice_pairs,
            indice_num,
            int(num_act_out),
            int(inverse),
            int(subm),
        )
        ctx.save_for_backward(features, filters, indice_pairs, indice_num)
        ctx.inverse = bool(inverse)
        ctx.subm = bool(subm)
        ctx.suffix = suffix
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, filters, indice_pairs, indice_num = ctx.saved_tensors
        function = getattr(
            _ops(), f"indice_conv_backward_{ctx.suffix}"
        )
        grad_features, grad_filters = function(
            features,
            filters,
            grad_output.contiguous(),
            indice_pairs,
            indice_num,
            int(ctx.inverse),
            int(ctx.subm),
        )
        return grad_features, grad_filters, None, None, None, None, None


def indice_conv(
    features,
    filters,
    indice_pairs,
    indice_num,
    num_act_out,
    inverse=False,
    subm=False,
    bias=None,
):
    """Apply an spconv-style sparse convolution on input features."""

    output = _IndiceConvFunction.apply(
        features,
        filters,
        indice_pairs,
        indice_num,
        num_act_out,
        inverse,
        subm,
    )
    return output if bias is None else output + bias


class _IndiceMaxPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indice_pairs, indice_num, num_act):
        suffix = _check_dtype(features)
        features = features.contiguous()
        indice_pairs = indice_pairs.contiguous()
        indice_num = indice_num.contiguous()
        function = getattr(_ops(), f"indice_maxpool_{suffix}")
        output = function(features, indice_pairs, indice_num, int(num_act))
        ctx.save_for_backward(features, output, indice_pairs, indice_num)
        ctx.suffix = suffix
        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, output, indice_pairs, indice_num = ctx.saved_tensors
        function = getattr(
            _ops(), f"indice_maxpool_backward_{ctx.suffix}"
        )
        grad_features = function(
            features,
            output,
            grad_output.contiguous(),
            indice_pairs,
            indice_num,
        )
        return grad_features, None, None, None


def indice_maxpool(features, indice_pairs, indice_num, num_act):
    """Apply spconv-style sparse max pooling on input features."""

    return _IndiceMaxPoolFunction.apply(
        features, indice_pairs, indice_num, num_act
    )


__all__ = [
    "get_indice_pairs",
    "indice_conv",
    "indice_maxpool",
]
