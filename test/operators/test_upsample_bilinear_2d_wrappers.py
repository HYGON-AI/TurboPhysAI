# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.operators import upsample_bilinear_2d as operator


def test_low_level_functions_delegate_to_c_extension(monkeypatch):
    calls = []
    marker = object()

    monkeypatch.setattr(
        operator,
        "upsample_bilinear_2d_forward",
        lambda *args: calls.append(("forward", args)) or marker,
    )
    monkeypatch.setattr(
        operator,
        "upsample_bilinear_2d_backward",
        lambda *args: calls.append(("backward", args)) or marker,
    )

    forward_args = ("input", [4, 4], False, None)
    assert operator.upsample_bilinear_2d_forward(*forward_args) is marker
    assert calls[-1] == ("forward", forward_args)

    backward_args = ("grad", [4, 4], (1, 1, 2, 2), False, None)
    assert operator.upsample_bilinear_2d_backward(*backward_args) is marker
    assert calls[-1] == ("backward", backward_args)


def test_interpolate_validation_errors():
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


def test_interpolate_recompute_scale_factor_uses_computed_output_size(monkeypatch):
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


def test_autograd_function_is_a_torch_function():
    assert issubclass(
        operator.UpSampleBilinear2dFunction,
        torch.autograd.Function,
    )
