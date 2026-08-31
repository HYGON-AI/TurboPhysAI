# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""MMCV adapter for the reusable HCU MDC implementation."""

from __future__ import annotations

from typing import Any


def is_supported_mdc_call(*args: Any, **kwargs: Any) -> bool:
    """Return whether the current MMCV call is covered by the HCU operator."""

    import torch

    def argument(name: str, position: int, default: Any = None) -> Any:
        return args[position] if len(args) > position else kwargs.get(name, default)

    input_tensor = argument("input", 0)
    offset = argument("offset", 1)
    mask = argument("mask", 2)
    weight = argument("weight", 3)
    bias = argument("bias", 4)
    tensors = (input_tensor, offset, mask, weight)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if not all(tensor.dim() == 4 for tensor in tensors):
        return False
    if not all(tensor.is_cuda for tensor in tensors):
        return False
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.numel() != weight.shape[0]
        or not bias.is_cuda
    ):
        return False

    groups = argument("groups", 8, 1)
    deform_groups = argument("deform_groups", 9, 1)
    if groups != 1 or not isinstance(deform_groups, int) or deform_groups < 1:
        return False
    if offset.dtype not in (torch.float16, torch.float32):
        return False

    def pair(value: Any) -> tuple[int, int] | None:
        if isinstance(value, int):
            return value, value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            if all(isinstance(item, int) for item in value):
                return value[0], value[1]
        return None

    stride = pair(argument("stride", 5, 1))
    padding = pair(argument("padding", 6, 0))
    dilation = pair(argument("dilation", 7, 1))
    if stride is None or padding is None or dilation is None:
        return False
    if any(value < 1 for value in (*stride, *dilation)):
        return False
    if any(value < 0 for value in padding):
        return False

    kernel_points = weight.shape[-2] * weight.shape[-1]
    output_size = tuple(
        (
            input_tensor.shape[dimension + 2]
            + 2 * padding[dimension]
            - dilation[dimension] * (weight.shape[dimension + 2] - 1)
            - 1
        )
        // stride[dimension]
        + 1
        for dimension in range(2)
    )
    return (
        weight.shape[0] > 0
        and weight.shape[1] == input_tensor.shape[1]
        and input_tensor.shape[1] % deform_groups == 0
        and offset.shape[0] == input_tensor.shape[0]
        and mask.shape[0] == input_tensor.shape[0]
        and offset.shape[1] == 2 * deform_groups * kernel_points
        and mask.shape[1] == deform_groups * kernel_points
        and tuple(offset.shape[-2:]) == output_size
        and tuple(mask.shape[-2:]) == output_size
    )


def modulated_deform_conv2d(*args: Any, **kwargs: Any) -> Any:
    """Call the HCU implementation with MMCV's public argument contract."""

    from ....operators.modulated_deform_conv import modulated_deform_conv2d

    return modulated_deform_conv2d(*args, **kwargs)


__all__ = ["is_supported_mdc_call", "modulated_deform_conv2d"]
