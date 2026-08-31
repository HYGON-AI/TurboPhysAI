# Derived from PyTorch: torch/nn/functional.py (interpolate).
# PyTorch v2.7.1, commit e2d141dbde55c2a4370fac5165b0561b6af4798b.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from turbo_physai.ops import upsample_bilinear_2d_forward, upsample_bilinear_2d_backward
from typing import Callable, List, Optional, Tuple, Union

Tensor = torch.Tensor

class UpSampleBilinear2dFunction(Function):
    @staticmethod
    def forward(ctx, input, output_size, align_corners, scale_factors):
        ctx.output_size = output_size
        ctx.input_size = input.shape
        ctx.align_corners = align_corners
        ctx.scale_factors = scale_factors

        output = upsample_bilinear_2d_forward(input, output_size, align_corners, scale_factors)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = upsample_bilinear_2d_backward(grad_output, ctx.output_size, ctx.input_size, ctx.align_corners, ctx.scale_factors)
        return grad_input, None, None, None

# 目前只针对特定size优化，其他size可能会coredump，请注意
def interpolate(input: Tensor,
                size: Optional[int] = None,
                scale_factor: Optional[List[float]] = None,
                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> Tensor:
    if mode != 'bilinear':
        raise ValueError(f"Only 'bilinear' mode is supported, but got '{mode}'")
    if antialias:
        raise ValueError("Antialias is not supported for 'bilinear' mode currently.")
    if input.dim() != 4:
        raise ValueError(f"bilinear Only 4D input tensors are supported, but got {input.dim()}D tensor.")
    
    if align_corners is not None and align_corners:
        raise ValueError("align_corners=True is not supported for upsampling bilinear mode now !")
    
    if align_corners is None:
        align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."

                )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is not None and recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        assert scale_factors is not None
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()))
                for i in range(dim)
            ]
        elif torch.jit.is_scripting():
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                           for i in range(dim)]
        else:
            output_size = [
                _sym_int(input.size(i + 2) * scale_factors[i])
                for i in range(dim)
            ]
        scale_factors = None

    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None, "align_corners option is required for upsampling bilinear"
        return UpSampleBilinear2dFunction.apply(input, output_size, align_corners, scale_factors)
    else:
        raise NotImplementedError(f"Only 4D input with 'bilinear' mode is supported, but got {input.dim()}D input with mode '{mode}'")
