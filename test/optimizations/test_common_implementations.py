# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.optimizations.common.mmdet3d.gaussian import gaussian_2d
from turbo_physai.optimizations.common.mmdet3d import sparse_conv
from turbo_physai.optimizations.common.mmdet3d.sparse_tensor import sparity


@pytest.mark.parametrize("shape,sigma", [((3, 3), 0.5), ((4, 6), 1.25)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gaussian_2d_matches_reference_formula(shape, sigma, dtype):
    actual = gaussian_2d(shape, sigma=sigma, dtype=dtype)
    middle_y, middle_x = [(size - 1.0) / 2.0 for size in shape]
    y = torch.arange(-middle_y, middle_y + 1, dtype=dtype).unsqueeze(1)
    x = torch.arange(-middle_x, middle_x + 1, dtype=dtype).unsqueeze(0)
    expected = torch.exp(-(x.square() + y.square()) / (2 * sigma * sigma))
    expected = torch.where(
        expected < torch.finfo(dtype).eps * expected.max(),
        0,
        expected,
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "spatial_shape,batch_size,rows,expected",
    [
        ((2, 3, 4), 2, 12, 0.25),
        ((4, 5, 6), 1, 0, 0.0),
        ((1, 1, 1), 4, 2, 0.5),
    ],
)
def test_sparse_tensor_sparity(spatial_shape, batch_size, rows, expected):
    tensor = type(
        "SparseTensor",
        (),
        {
            "spatial_shape": spatial_shape,
            "batch_size": batch_size,
            "indices": torch.empty((rows, 4), dtype=torch.int32),
        },
    )()
    assert sparity.__get__(tensor) == pytest.approx(expected)


def test_sparse_convolution_output_shapes_and_invalid_parameter_boundary():
    assert sparse_conv._conv_output_size(
        [8, 10, 12], [3, 3, 3], [2, 1, 1], [1, 1, 1], [1, 1, 1]
    ) == [4, 10, 12]
    assert sparse_conv._deconv_output_size(
        [4, 5, 6], [3, 3, 3], [2, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]
    ) == [8, 5, 6]
    with pytest.raises(ValueError, match="kernel_size < 0"):
        sparse_conv._deconv_output_size(
            [4], [-1], [1], [0], [1], [0]
        )


def test_indice_frontend_rejects_stride_and_dilation_combination():
    indices = torch.zeros((1, 4), dtype=torch.int32)
    with pytest.raises(
        ValueError, match="stride and dilation cannot both exceed one"
    ):
        sparse_conv.get_indice_pairs(
            indices,
            1,
            [4, 4, 4],
            ksize=3,
            stride=2,
            padding=1,
            dilation=2,
        )
