# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-independent PyTorch reference implementations for operator tests."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


def _pair(value: int | Iterable[int]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    first, second = value
    return int(first), int(second)


def _normalize_grid(coordinate: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(coordinate)
    return coordinate.mul(2.0 / (size - 1)).sub(1.0)


def modulated_deform_conv2d_reference(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int | Iterable[int] = 1,
    padding: int | Iterable[int] = 0,
    dilation: int | Iterable[int] = 1,
    groups: int = 1,
    deform_groups: int = 1,
) -> torch.Tensor:
    """Small differentiable MDC reference independent of MMCV.

    The offset layout follows MMCV/hipDNN: each deformable group stores
    interleaved vertical and horizontal offsets for every kernel point as
    ``[y0, x0, y1, x1, ...]``. This implementation favors readability over
    performance and is intended only for fixed-size correctness tests.
    """

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)
    batch, in_channels, input_h, input_w = input.shape
    out_channels, weight_channels, kernel_h, kernel_w = weight.shape
    kernel_points = kernel_h * kernel_w
    output_h, output_w = offset.shape[-2:]

    if in_channels % groups or out_channels % groups:
        raise ValueError("input and output channels must be divisible by groups")
    if in_channels % deform_groups:
        raise ValueError("input channels must be divisible by deform_groups")
    if weight_channels != in_channels // groups:
        raise ValueError("weight shape is inconsistent with groups")
    if offset.shape[1] != deform_groups * 2 * kernel_points:
        raise ValueError("offset channel count is inconsistent with deform_groups")
    if mask.shape[1] != deform_groups * kernel_points:
        raise ValueError("mask channel count is inconsistent with deform_groups")

    offset = offset.reshape(
        batch, deform_groups, kernel_points, 2, output_h, output_w
    )
    mask = mask.reshape(
        batch, deform_groups, kernel_points, output_h, output_w
    )
    channels_per_deform_group = in_channels // deform_groups
    output_y = torch.arange(output_h, device=input.device, dtype=input.dtype)
    output_x = torch.arange(output_w, device=input.device, dtype=input.dtype)
    base_y, base_x = torch.meshgrid(output_y, output_x, indexing="ij")
    sampled_by_group = []
    for deform_group in range(deform_groups):
        channel_start = deform_group * channels_per_deform_group
        channel_end = channel_start + channels_per_deform_group
        group_input = input[:, channel_start:channel_end]
        sampled_points = []
        for kernel_y in range(kernel_h):
            for kernel_x in range(kernel_w):
                point = kernel_y * kernel_w + kernel_x
                sample_y = (
                    base_y * stride_h
                    - pad_h
                    + kernel_y * dilation_h
                    + offset[:, deform_group, point, 0]
                )
                sample_x = (
                    base_x * stride_w
                    - pad_w
                    + kernel_x * dilation_w
                    + offset[:, deform_group, point, 1]
                )
                grid = torch.stack(
                    (
                        _normalize_grid(sample_x, input_w),
                        _normalize_grid(sample_y, input_h),
                    ),
                    dim=-1,
                )
                sampled = F.grid_sample(
                    group_input,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                sampled_points.append(
                    sampled * mask[:, deform_group, point].unsqueeze(1)
                )
        sampled_by_group.append(torch.stack(sampled_points, dim=2))

    sampled = torch.cat(sampled_by_group, dim=1)
    sampled = sampled.reshape(
        batch,
        groups,
        in_channels // groups,
        kernel_points,
        output_h,
        output_w,
    )
    grouped_weight = weight.reshape(
        groups, out_channels // groups, in_channels // groups, kernel_points
    )
    output = torch.einsum("ngckhw,gock->ngohw", sampled, grouped_weight)
    output = output.reshape(batch, out_channels, output_h, output_w)
    if bias is not None:
        output = output + bias.reshape(1, -1, 1, 1)
    return output
