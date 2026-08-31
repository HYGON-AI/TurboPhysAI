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

"""Canonical MMDetection3D sparse-convolution helpers."""


def _ops():
    from turbo_physai import ops

    return ops


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
