# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from test.reference_operators import modulated_deform_conv2d_reference


pytestmark = pytest.mark.hcu


def _require_hcu_operator():
    if not torch.cuda.is_available():
        pytest.skip("a real HCU device is required")
    from turbo_physai import modulated_deform_conv2d

    return modulated_deform_conv2d


def test_forward_and_supported_gradients_match_independent_reference():
    operator = _require_hcu_operator()
    torch.manual_seed(42)
    input = torch.randn(1, 4, 5, 6, requires_grad=True)
    weight = (torch.randn(4, 4, 3, 3) * 0.01).requires_grad_()
    offset = (torch.rand(1, 18, 5, 6) - 0.5).requires_grad_()
    mask = torch.sigmoid(torch.randn(1, 9, 5, 6)).requires_grad_()
    reference = modulated_deform_conv2d_reference(
        input, offset, mask, weight, padding=1
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    reference_input_grad, reference_weight_grad = torch.autograd.grad(
        reference, (input, weight), grad_output, retain_graph=True
    )

    hcu_input = input.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    hcu_weight = weight.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    hcu_offset = offset.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    hcu_mask = mask.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    actual = operator(
        hcu_input,
        hcu_offset,
        hcu_mask,
        hcu_weight,
        None,
        1,
        1,
        1,
        1,
    )
    actual.backward(grad_output.cuda())

    torch.testing.assert_close(actual.cpu(), reference.detach(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        hcu_input.grad.cpu(), reference_input_grad, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        hcu_weight.grad.cpu(), reference_weight_grad, rtol=1e-1, atol=1e-1
    )
    # The current public operator contract intentionally does not propagate
    # offset or mask gradients. Record that limitation instead of silently
    # treating those gradients as validated.
    assert hcu_offset.grad is None or torch.count_nonzero(hcu_offset.grad) == 0
    assert hcu_mask.grad is None or torch.count_nonzero(hcu_mask.grad) == 0


def test_bias_is_reported_as_unsupported():
    operator = _require_hcu_operator()
    input = torch.zeros(1, 4, 3, 3, device="cuda")
    offset = torch.zeros(1, 18, 3, 3, device="cuda")
    mask = torch.ones(1, 9, 3, 3, device="cuda")
    weight = torch.zeros(4, 4, 3, 3, device="cuda")
    bias = torch.zeros(4, device="cuda")
    with pytest.raises(NotImplementedError, match="does not support with bias"):
        operator(input, offset, mask, weight, bias, 1, 1, 1, 1)
