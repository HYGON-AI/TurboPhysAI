# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


torch = pytest.importorskip("torch")


@pytest.fixture
def native_extension(monkeypatch):
    extension = types.ModuleType("turbo_physai._C")
    extension.upsample_bilinear_2d_forward = Mock(name="native_forward")
    extension.upsample_bilinear_2d_backward = Mock(name="native_backward")
    monkeypatch.setitem(sys.modules, "turbo_physai._C", extension)
    return extension


@pytest.fixture
def operator(native_extension):
    # Load fresh bindings without a built extension or cached operator module.
    path = (
        Path(__file__).resolve().parents[2]
        / "turbo_physai/operators/upsample_bilinear_2d.py"
    )
    spec = importlib.util.spec_from_file_location(
        "turbo_physai.operators.upsample_bilinear_2d", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_low_level_functions_are_bound_to_c_extension(operator, native_extension):
    assert (
        operator.upsample_bilinear_2d_forward
        is native_extension.upsample_bilinear_2d_forward
    )
    assert (
        operator.upsample_bilinear_2d_backward
        is native_extension.upsample_bilinear_2d_backward
    )


@pytest.mark.parametrize(
    "options, output_size, scale_factors, output_shape",
    [
        ({"size": (4, 6)}, (4, 6), None, (4, 6)),
        ({"size": 4}, [4, 4], None, (4, 4)),
        ({"scale_factor": (2.0, 3.0)}, None, (2.0, 3.0), (4, 9)),
        ({"scale_factor": 2.0}, None, [2.0, 2.0], (4, 6)),
        (
            {"scale_factor": 2.0, "recompute_scale_factor": True},
            [4, 6],
            None,
            (4, 6),
        ),
    ],
    ids=["tuple-size", "scalar-size", "tuple-scale", "scalar-scale", "recompute"],
)
def test_interpolate_delegates_forward_and_backward(
    monkeypatch, operator, options, output_size, scale_factors, output_shape
):
    input = torch.zeros(1, 2, 2, 3, requires_grad=True)
    forward_output = torch.full((1, 2, *output_shape), 7.0)
    backward_output = torch.full_like(input, 3.0)
    forward = Mock(return_value=forward_output)
    backward = Mock(return_value=backward_output)
    monkeypatch.setattr(operator, "upsample_bilinear_2d_forward", forward)
    monkeypatch.setattr(operator, "upsample_bilinear_2d_backward", backward)

    output = operator.interpolate(input, mode="bilinear", **options)

    forward.assert_called_once_with(input, output_size, False, scale_factors)
    torch.testing.assert_close(output, forward_output)
    backward.assert_not_called()

    output.sum().backward()

    backward.assert_called_once()
    assert backward.call_args.kwargs == {}
    grad_output, *backward_args = backward.call_args.args
    torch.testing.assert_close(grad_output, torch.ones_like(forward_output))
    assert backward_args == [output_size, input.shape, False, scale_factors]
    torch.testing.assert_close(input.grad, backward_output)


def test_interpolate_validation_errors(operator):
    input = torch.zeros(1, 1, 2, 2)

    with pytest.raises(ValueError, match="Only 'bilinear'"):
        operator.interpolate(input, size=(4, 4), mode="nearest")
    with pytest.raises(ValueError, match="Antialias"):
        operator.interpolate(input, size=(4, 4), mode="bilinear", antialias=True)
    with pytest.raises(ValueError, match="Only 4D"):
        operator.interpolate(torch.zeros(1, 1, 2), size=(4,), mode="bilinear")
    with pytest.raises(ValueError, match="align_corners=True"):
        operator.interpolate(
            input,
            size=(4, 4),
            mode="bilinear",
            align_corners=True,
        )
    with pytest.raises(ValueError, match="only one of size or scale_factor"):
        operator.interpolate(
            input,
            size=(4, 4),
            scale_factor=2,
            mode="bilinear",
        )
    with pytest.raises(ValueError, match="either size or scale_factor"):
        operator.interpolate(input, mode="bilinear")


def test_interpolate_recompute_scale_factor_uses_computed_output_size(
    monkeypatch, operator
):
    calls = []

    def forward(input, output_size, align_corners, scale_factors):
        calls.append((output_size, align_corners, scale_factors))
        return input.clone()

    monkeypatch.setattr(operator, "upsample_bilinear_2d_forward", forward)

    input = torch.zeros(1, 1, 2, 3)
    output = operator.interpolate(
        input,
        scale_factor=2,
        mode="bilinear",
        recompute_scale_factor=True,
    )

    torch.testing.assert_close(output, input)
    assert calls == [([4, 6], False, None)]


def test_autograd_function_is_a_torch_function(operator):
    assert issubclass(
        operator.UpSampleBilinear2dFunction,
        torch.autograd.Function,
    )
