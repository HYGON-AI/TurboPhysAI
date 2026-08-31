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
    (
        reference_input_grad,
        reference_offset_grad,
        reference_mask_grad,
        reference_weight_grad,
    ) = torch.autograd.grad(reference, (input, offset, mask, weight), grad_output)

    hcu_input = (
        input.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_weight = (
        weight.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_offset = (
        offset.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_mask = (
        mask.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
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
    torch.testing.assert_close(
        hcu_offset.grad.cpu(), reference_offset_grad, rtol=1e-1, atol=1e-1
    )
    torch.testing.assert_close(
        hcu_mask.grad.cpu(), reference_mask_grad, rtol=1e-1, atol=1e-1
    )


def test_bias_forward_and_gradient_match_reference():
    operator = _require_hcu_operator()
    torch.manual_seed(43)
    input = torch.randn(1, 4, 5, 6, requires_grad=True)
    weight = (torch.randn(4, 4, 3, 3) * 0.01).requires_grad_()
    bias = (torch.randn(4) * 0.01).requires_grad_()
    offset = torch.rand(1, 18, 5, 6) - 0.5
    mask = torch.sigmoid(torch.randn(1, 9, 5, 6))
    reference = modulated_deform_conv2d_reference(
        input, offset, mask, weight, bias, padding=1
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    reference_grads = torch.autograd.grad(reference, (input, weight, bias), grad_output)

    hcu_input = (
        input.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_weight = (
        weight.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_bias = bias.detach().cuda().requires_grad_()
    actual = operator(
        hcu_input,
        offset.cuda().to(memory_format=torch.channels_last),
        mask.cuda().to(memory_format=torch.channels_last),
        hcu_weight,
        hcu_bias,
        1,
        1,
        1,
        1,
    )
    actual.backward(grad_output.cuda())

    torch.testing.assert_close(actual.cpu(), reference.detach(), rtol=1e-3, atol=1e-3)
    for actual_grad, reference_grad in zip(
        (hcu_input.grad, hcu_weight.grad, hcu_bias.grad), reference_grads
    ):
        torch.testing.assert_close(
            actual_grad.cpu(), reference_grad, rtol=1e-1, atol=1e-1
        )


def test_multiple_deform_groups_match_reference():
    operator = _require_hcu_operator()
    torch.manual_seed(44)
    input = torch.randn(1, 4, 5, 6, requires_grad=True)
    weight = (torch.randn(4, 4, 3, 3) * 0.01).requires_grad_()
    offset = (torch.rand(1, 36, 5, 6) - 0.5).requires_grad_()
    mask = torch.sigmoid(torch.randn(1, 18, 5, 6)).requires_grad_()
    reference = modulated_deform_conv2d_reference(
        input,
        offset,
        mask,
        weight,
        padding=1,
        deform_groups=2,
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    reference_grads = torch.autograd.grad(
        reference, (input, offset, mask, weight), grad_output
    )

    hcu_tensors = [
        tensor.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
        for tensor in (input, offset, mask, weight)
    ]
    actual = operator(
        *hcu_tensors,
        None,
        1,
        1,
        1,
        1,
        2,
    )
    actual.backward(grad_output.cuda())

    torch.testing.assert_close(actual.cpu(), reference.detach(), rtol=1e-3, atol=1e-3)
    for tensor, reference_grad in zip(hcu_tensors, reference_grads):
        torch.testing.assert_close(
            tensor.grad.cpu(), reference_grad, rtol=1e-1, atol=1e-1
        )


def test_fp16_forward_and_gradients_match_reference():
    operator = _require_hcu_operator()
    torch.manual_seed(45)
    input = torch.randn(1, 4, 5, 6).half().float().requires_grad_()
    weight = (torch.randn(4, 4, 3, 3) * 0.01).half().float().requires_grad_()
    offset = (torch.rand(1, 18, 5, 6) - 0.5).half().float().requires_grad_()
    mask = torch.sigmoid(torch.randn(1, 9, 5, 6)).half().float().requires_grad_()
    reference = modulated_deform_conv2d_reference(
        input, offset, mask, weight, padding=1
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    reference_grads = torch.autograd.grad(
        reference, (input, offset, mask, weight), grad_output
    )

    hcu_tensors = [
        tensor.detach()
        .half()
        .to(memory_format=torch.channels_last)
        .cuda()
        .requires_grad_()
        for tensor in (input, offset, mask, weight)
    ]
    actual = operator(
        *hcu_tensors,
        None,
        1,
        1,
        1,
        1,
    )
    actual.backward(grad_output.half().cuda())

    torch.testing.assert_close(
        actual.float().cpu(), reference.detach(), rtol=2e-2, atol=2e-2
    )
    for tensor, reference_grad in zip(hcu_tensors, reference_grads):
        torch.testing.assert_close(
            tensor.grad.float().cpu(), reference_grad, rtol=1e-1, atol=1e-1
        )


def test_required_gradients_skips_offset_and_mask_outputs():
    operator = _require_hcu_operator()
    torch.manual_seed(46)
    input = torch.randn(1, 4, 5, 6, requires_grad=True)
    weight = (torch.randn(4, 4, 3, 3) * 0.01).requires_grad_()
    offset = (torch.rand(1, 18, 5, 6) - 0.5).requires_grad_()
    mask = torch.sigmoid(torch.randn(1, 9, 5, 6)).requires_grad_()
    reference = modulated_deform_conv2d_reference(
        input, offset, mask, weight, padding=1
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    reference_input_grad, reference_weight_grad = torch.autograd.grad(
        reference, (input, weight), grad_output
    )

    hcu_input = (
        input.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_offset = (
        offset.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_mask = (
        mask.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
    hcu_weight = (
        weight.detach().to(memory_format=torch.channels_last).cuda().requires_grad_()
    )
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
        required_gradients=("input", "weight"),
    )
    actual.backward(grad_output.cuda())

    torch.testing.assert_close(
        hcu_input.grad.cpu(), reference_input_grad, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        hcu_weight.grad.cpu(), reference_weight_grad, rtol=1e-1, atol=1e-1
    )
    assert hcu_offset.grad is None
    assert hcu_mask.grad is None
