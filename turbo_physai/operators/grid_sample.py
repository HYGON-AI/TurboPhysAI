# Derived from PyTorch: torch/nn/functional.py (grid_sample).
# PyTorch v2.7.1, commit e2d141dbde55c2a4370fac5165b0561b6af4798b.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from turbo_physai.ops import grid_sample_forward, grid_sample_backward
from typing import Callable, List, Optional, Tuple, Union

Tensor = torch.Tensor

class GridSampleFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, mode, padding_mode, align_corners=None):
        ctx.mode = mode
        ctx.padding_mode = padding_mode
        ctx.align_corners = align_corners
        ctx.save_for_backward(input, grid)
        output = grid_sample_forward(input, grid, mode, padding_mode, align_corners)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        output_mask = [True, True]
        grad_input, grad_grid = grid_sample_backward(grad_output, input, grid, ctx.mode, ctx.padding_mode, ctx.align_corners, output_mask)
        return grad_input, grad_grid, None, None, None


def grid_sample(
    input: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: Optional[bool] = None,
) -> Tensor:
    # 目前优化版本只优化了2d bilinear的场景，其他场景不支持
    if mode != "bilinear":
        raise ValueError(
            "grid_sample(): expected mode to be "
            "'bilinear', but got: '{}'".format(mode)
        )
    if input.dim() != 4 or grid.dim() != 4:
        raise ValueError(
            "grid_sample(): only support 4D input and grid (got {}D and {}D)".format(input.dim(), grid.dim())
        )
    if padding_mode != "zeros" and padding_mode != "border" and padding_mode != "reflection":
        raise ValueError(
            "grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )

    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if align_corners is None:
        warnings.warn(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    return GridSampleFunction.apply(input, grid, mode_enum, padding_mode_enum, align_corners)
