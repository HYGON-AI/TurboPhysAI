# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.operators import sparse_conv as operator


def _require_hcu():
    if not torch.cuda.is_available():
        pytest.skip("a real HCU device is required")
    import turbo_physai._C  # noqa: F401


def test_get_indice_pairs_dispatches_by_rank(monkeypatch):
    calls = []
    marker = object()

    class Extension:
        @staticmethod
        def get_indice_pairs_3d(*args):
            calls.append(args)
            return marker

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    indices = torch.zeros((2, 4), dtype=torch.int32)
    result = operator.get_indice_pairs(
        indices,
        batch_size=1,
        spatial_shape=[4, 5, 6],
        ksize=3,
        stride=1,
        padding=1,
    )

    assert result is marker
    assert calls[0][0] is indices
    assert calls[0][1:] == (
        1,
        [4, 5, 6],
        [4, 5, 6],
        [3, 3, 3],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 0],
        0,
        0,
    )


def test_get_indice_pairs_dispatches_grid_path(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def get_indice_pairs_grid_2d(*args):
            calls.append(args)
            return "grid"

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    result = operator.get_indice_pairs(
        torch.zeros((2, 3), dtype=torch.int32),
        batch_size=1,
        spatial_shape=[4, 4],
        grid="grid",
    )

    assert result == "grid"
    assert calls[0][1] == "grid"


def test_get_indice_pairs_rejects_stride_and_dilation_together():
    with pytest.raises(ValueError, match="stride and dilation"):
        operator.get_indice_pairs(
            torch.zeros((1, 4), dtype=torch.int32),
            batch_size=1,
            spatial_shape=[4, 4, 4],
            stride=2,
            dilation=2,
        )


def test_indice_conv_dispatch_bias_and_backward(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def indice_conv_fp32(
            features,
            filters,
            indice_pairs,
            indice_num,
            num_act_out,
            inverse,
            subm,
        ):
            calls.append(
                (
                    "forward",
                    features.dtype,
                    num_act_out,
                    inverse,
                    subm,
                )
            )
            return torch.ones((2, 4), dtype=features.dtype)

        @staticmethod
        def indice_conv_backward_fp32(
            features,
            filters,
            grad_output,
            indice_pairs,
            indice_num,
            inverse,
            subm,
        ):
            calls.append(("backward", grad_output.is_contiguous()))
            return torch.ones_like(features), torch.ones_like(filters)

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    features = torch.randn(2, 3, requires_grad=True)
    filters = torch.randn(4, 3, requires_grad=True)
    indice_pairs = torch.zeros((1, 1, 1), dtype=torch.int32)
    indice_num = torch.ones(1, dtype=torch.int32)
    bias = torch.arange(4, dtype=torch.float32)

    output = operator.indice_conv(
        features,
        filters,
        indice_pairs,
        indice_num,
        num_act_out=4,
        bias=bias,
    )
    output.sum().backward()

    assert calls == [
        ("forward", torch.float32, 4, 0, 0),
        ("backward", True),
    ]
    torch.testing.assert_close(output, torch.ones((2, 4)) + bias)
    torch.testing.assert_close(features.grad, torch.ones_like(features))
    torch.testing.assert_close(filters.grad, torch.ones_like(filters))


def test_indice_maxpool_dispatch_and_backward(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def indice_maxpool_fp32(features, indice_pairs, indice_num, num_act):
            calls.append(("forward", num_act))
            return torch.ones((2, 4), dtype=features.dtype)

        @staticmethod
        def indice_maxpool_backward_fp32(
            features,
            output,
            grad_output,
            indice_pairs,
            indice_num,
        ):
            calls.append(("backward", grad_output.is_contiguous()))
            return torch.ones_like(features)

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    features = torch.randn(2, 3, requires_grad=True)
    indice_pairs = torch.zeros((1, 1, 1), dtype=torch.int32)
    indice_num = torch.ones(1, dtype=torch.int32)

    output = operator.indice_maxpool(features, indice_pairs, indice_num, 4)
    output.sum().backward()

    assert calls == [("forward", 4), ("backward", True)]
    torch.testing.assert_close(features.grad, torch.ones_like(features))


@pytest.mark.hcu
def test_indice_conv_matches_reference_forward_and_backward():
    _require_hcu()
    torch.manual_seed(123)
    device = "cuda"
    in_channels, out_channels = 4, 3

    features = torch.randn(4, in_channels, device=device, requires_grad=True)
    filters = torch.randn(
        2, 1, 1, in_channels, out_channels, device=device, requires_grad=True
    )
    bias = torch.randn(out_channels, device=device, requires_grad=True)
    indice_pairs = torch.tensor(
        [
            [[0, 1], [0, 1]],
            [[2, 3], [0, 1]],
        ],
        dtype=torch.int32,
        device=device,
    )
    indice_num = torch.tensor([2, 2], dtype=torch.int32, device=device)

    output = operator.indice_conv(
        features,
        filters,
        indice_pairs,
        indice_num,
        num_act_out=2,
        bias=bias,
    )

    filters_2d = filters.detach().reshape(2, in_channels, out_channels)
    expected = torch.zeros(2, out_channels, device=device)
    for kernel, count in enumerate(indice_num.tolist()):
        for pair in range(count):
            in_index = int(indice_pairs[kernel, 0, pair])
            out_index = int(indice_pairs[kernel, 1, pair])
            expected[out_index] += features.detach()[in_index] @ filters_2d[kernel]
    expected += bias.detach()
    torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    expected_feature_grad = torch.zeros_like(features)
    expected_filter_grad = torch.zeros_like(filters)
    for kernel, count in enumerate(indice_num.tolist()):
        for pair in range(count):
            in_index = int(indice_pairs[kernel, 0, pair])
            out_index = int(indice_pairs[kernel, 1, pair])
            expected_feature_grad[in_index] += (
                grad_output[out_index] @ filters_2d[kernel].t()
            )
            expected_filter_grad[kernel, 0, 0] += torch.outer(
                features.detach()[in_index], grad_output[out_index]
            )

    torch.testing.assert_close(
        features.grad, expected_feature_grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        filters.grad, expected_filter_grad, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        bias.grad, grad_output.sum(dim=0), rtol=1e-4, atol=1e-4
    )


@pytest.mark.hcu
def test_indice_maxpool_matches_reference_forward_and_backward():
    _require_hcu()
    torch.manual_seed(321)
    device = "cuda"

    features = torch.tensor(
        [
            [1.0, 5.0, -2.0],
            [4.0, 5.0, -1.0],
            [3.0, -1.0, 6.0],
            [2.0, 7.0, 2.0],
        ],
        device=device,
        requires_grad=True,
    )
    indice_pairs = torch.tensor(
        [
            [[0, 2], [0, 1]],
            [[1, 3], [0, 1]],
        ],
        dtype=torch.int32,
        device=device,
    )
    indice_num = torch.tensor([2, 2], dtype=torch.int32, device=device)

    output = operator.indice_maxpool(features, indice_pairs, indice_num, 2)

    expected = torch.zeros(2, features.shape[1], device=device)
    for kernel, count in enumerate(indice_num.tolist()):
        for pair in range(count):
            in_index = int(indice_pairs[kernel, 0, pair])
            out_index = int(indice_pairs[kernel, 1, pair])
            expected[out_index] = torch.maximum(
                expected[out_index], features.detach()[in_index]
            )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    expected_feature_grad = torch.zeros_like(features)
    for kernel, count in enumerate(indice_num.tolist()):
        for pair in range(count):
            in_index = int(indice_pairs[kernel, 0, pair])
            out_index = int(indice_pairs[kernel, 1, pair])
            matched = features.detach()[in_index] == expected[out_index]
            expected_feature_grad[in_index] += matched * grad_output[out_index]

    torch.testing.assert_close(
        features.grad, expected_feature_grad, rtol=1e-5, atol=1e-5
    )


def test_indice_conv_rejects_mismatched_dtypes():
    with pytest.raises(TypeError, match="same dtype"):
        operator.indice_conv(
            torch.zeros(2, 3),
            torch.zeros(4, 3, dtype=torch.float16),
            torch.zeros((1, 1, 1), dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            num_act_out=4,
        )
