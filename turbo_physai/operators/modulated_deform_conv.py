# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
import math
from typing import Optional, Tuple, Union
import os
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import hipdnn


shape_dict = {}

def build_fprop_graph(input, offset, weight, mask, stride, padding, dilation):
    if input.dtype == torch.float32:
        hipdnn_dtype = hipdnn.data_type.FLOAT
    elif input.dtype == torch.float16:
        hipdnn_dtype = hipdnn.data_type.HALF
    else:
        raise ValueError(
                f"miopen deformer conv not support dtyp {input.dtype}")

    graph = hipdnn.pygraph(
        name="deform_convolution",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn_dtype,
    )

    input_h = graph.tensor_like(input.detach())
    offset_h = graph.tensor_like(offset.detach())
    weight_h = graph.tensor_like(weight.detach())
    mask_h = graph.tensor_like(mask.detach())

    out = graph.deform_conv_fprop(
        image=input_h,
        offset=offset_h,
        weight=weight_h,
        mask=mask_h,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="deform_conv_fprop",
    )
    out.set_output(True).set_data_type(hipdnn_dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()

    return (graph, input_h, offset_h, weight_h, mask_h, out)

def build_wrw_graph(input, offset, weight, mask, grad_output, stride, padding, dilation):
    if input.dtype == torch.float32:
        hipdnn_dtype = hipdnn.data_type.FLOAT
    elif input.dtype == torch.float16:
        hipdnn_dtype = hipdnn.data_type.HALF
    else:
        raise ValueError(
                f"miopen deformer conv not support dtyp {input.dtype}")

    graph = hipdnn.pygraph(
        name="deform_convolution_wrw",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn_dtype,
    )

    input_h = graph.tensor_like(input.detach())
    offset_h = graph.tensor_like(offset.detach())
    grad_output_h = graph.tensor_like(grad_output.detach())
    mask_h = graph.tensor_like(mask.detach())

    dw = graph.deform_conv_wgrad(
        image=input_h,
        offset=offset_h,
        loss=grad_output_h,
        mask=mask_h,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="deform_conv_wgrad",
    )
    dw.set_dim(weight.shape).set_output(True).set_data_type(hipdnn_dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()

    return (graph, input_h, offset_h, mask_h, grad_output_h, dw)


def build_dx_graph(input, offset, weight, mask, grad_output, stride, padding, dilation):
    hipdnn_dtype = hipdnn.data_type.FLOAT
    if input.dtype == torch.float32:
        hipdnn_dtype = hipdnn.data_type.FLOAT
    elif input.dtype == torch.float16:
        hipdnn_dtype = hipdnn.data_type.HALF
    else:
        raise ValueError(
                f"miopen deformer conv not support dtyp {input.dtype}")

    graph = hipdnn.pygraph(
        name="deform_convolution_bwd",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn_dtype,
    )

    offset_h = graph.tensor_like(offset.detach())
    weight_h = graph.tensor_like(weight.detach())
    mask_h = graph.tensor_like(mask.detach())
    grad_output_h = graph.tensor_like(grad_output.detach())
    input_h = graph.tensor_like(input.detach())

    dx, doffset, dmask = graph.deform_conv_dgrad(
        loss=grad_output_h,
        offset=offset_h,
        filter=weight_h,
        mask=mask_h,
        image=input_h,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="deform_conv_dgrad",
    )
    dx.set_dim(input.shape).set_output(True).set_data_type(hipdnn_dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()

    return (graph, offset_h, weight_h, mask_h, grad_output_h, dx, doffset, dmask)


class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    def _miopen_output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0),)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        output_size += (channels,)
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size

    @staticmethod
    def _hcu_forward(ctx, input, offset, mask, weight, bias):
        fp16_mode = os.getenv('ENABLE_MIOPEN_DEFORMER_CONV_FP16', 'False').lower() == 'true'
        origin_type = input.dtype
        if fp16_mode and (origin_type == torch.float32):
            input = input.to(dtype=torch.float16)
            offset = offset.to(dtype=torch.float16)
            mask = mask.to(dtype=torch.float16)
            weight = weight.to(dtype=torch.float16)
        
        output = input.new_empty([
            int(i) for i in ModulatedDeformConv2dFunction._miopen_output_size(
                ctx, input, weight)
        ])
        if input.shape not in shape_dict:
            graph, input_h, offset_h, weight_h, mask_h, out = build_fprop_graph(input, offset, weight, mask, ctx.stride, ctx.padding, ctx.dilation)
            shape_dict[input.shape] = [graph, input_h, offset_h, weight_h, mask_h, out]

        graph, input_h, offset_h, weight_h, mask_h, out = shape_dict[input.shape]

        variant_pack = {
            input_h: input.data_ptr(),
            offset_h: offset.data_ptr(),
            weight_h: weight.data_ptr(),
            mask_h: mask.data_ptr(),
            out: output.data_ptr(),
        }
        workspace = torch.zeros(graph.get_workspace_size(), dtype=torch.uint8, device=input.device)

        graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())
        if fp16_mode and (origin_type == torch.float32):
            output = output.to(dtype=torch.float32)

        return output.permute(0, 3, 1, 2)

    @staticmethod
    def _hcu_backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        fp16_mode = os.getenv('ENABLE_MIOPEN_DEFORMER_CONV_FP16', 'False').lower() == 'true'
        origin_type = input.dtype
        if fp16_mode and (origin_type == torch.float32):
            input = input.to(dtype=torch.float16)
            offset = offset.to(dtype=torch.float16)
            mask = mask.to(dtype=torch.float16)
            weight = weight.to(dtype=torch.float16)
            grad_output = grad_output.to(dtype=torch.float16)

        grad_input = torch.zeros_like(input)
        grad_offset = torch.zeros_like(offset)
        grad_mask = torch.zeros_like(mask)
        grad_weight = torch.zeros_like(weight)
        grad_output = grad_output.to(memory_format=torch.channels_last)

        shape_key = ','.join(map(str, input.shape)) + 'wrw'
        if shape_key not in shape_dict:
            graph, input_h, offset_h, mask_h, grad_output_h, dw = build_wrw_graph(input, offset, weight, mask, grad_output,
                                                                                  ctx.stride, ctx.padding, ctx.dilation)
            shape_dict[shape_key] = [graph, input_h, offset_h, mask_h, grad_output_h, dw]

        graph, input_h, offset_h, mask_h, grad_output_h, dw = shape_dict[shape_key]
        variant_pack = {
            input_h: input.data_ptr(),
            offset_h: offset.data_ptr(),
            grad_output_h: grad_output.data_ptr(),
            mask_h: mask.data_ptr(),
            dw: grad_weight.data_ptr(),
        }
        workspace = torch.zeros(graph.get_workspace_size(), dtype=torch.uint8, device=input.device)
        graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())

        shape_key = ','.join(map(str, input.shape)) + 'dx'
        if shape_key not in shape_dict:
            graph, offset_h, weight_h, mask_h, grad_output_h, dx, doffset, dmask = build_dx_graph(input, offset, weight, mask, grad_output, ctx.stride, ctx.padding, ctx.dilation)
            shape_dict[shape_key] = [graph, offset_h, weight_h, mask_h, grad_output_h, dx, doffset, dmask]

        graph, offset_h, weight_h, mask_h, grad_output_h, dx, doffset, dmask = shape_dict[shape_key]
        variant_pack = {
            grad_output_h: grad_output.data_ptr(),
            weight_h: weight.data_ptr(),
            offset_h: offset.data_ptr(),
            mask_h: mask.data_ptr(),
            dx: grad_input.data_ptr(),
            doffset: grad_offset.data_ptr(),
            dmask: grad_mask.data_ptr(),
        }
        workspace_dx = torch.zeros(graph.get_workspace_size(), dtype=torch.uint8, device=input.device)
        graph.exec(variant_pack=variant_pack, workspace=workspace_dx.data_ptr())

        if fp16_mode and (origin_type == torch.float32):
            grad_input = grad_input.to(dtype=torch.float32)
            grad_offset = grad_offset.to(dtype=torch.float32)
            grad_mask = grad_mask.to(dtype=torch.float32)
            grad_weight = grad_weight.to(dtype=torch.float32)

        # grad_offset, grad_mask的计算目前不支持，先站位；
        return (grad_input, grad_offset, grad_mask, grad_weight, None,
                None, None, None, None, None)

    @staticmethod
    def forward(ctx,
                input: torch.Tensor,
                offset: torch.Tensor,
                mask: torch.Tensor,
                weight: nn.Parameter,
                bias: Optional[nn.Parameter] = None,
                stride: int = 1,
                padding: int = 0,
                dilation: int = 1,
                groups: int = 1,
                deform_groups: int = 1) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f'Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead.')
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        # When pytorch version >= 1.6.0, amp is adopted for fp16 mode;
        # amp won't cast the type of model (float32), but "offset" is cast
        # to float16 by nn.Conv2d automatically, leading to the type
        # mismatch with input (when it is float32) or weight.
        # The flag for whether to use fp16 or amp is the type of "offset",
        # we cast weight and input to temporarily support fp16 and amp
        # whatever the pytorch version is.
        input = input.type_as(offset)
        weight = weight.type_as(input)
        bias = bias.type_as(input)  # type: ignore
        ctx.save_for_backward(input, offset, mask, weight, bias)
        ctx._bufs = [input.new_empty(0), input.new_empty(0)]
        if ctx.with_bias:
            raise NotImplementedError("ModulatedDeformConv2d does not support with bias yet.")
        return ModulatedDeformConv2dFunction._hcu_forward(
            ctx, input, offset, mask, weight, bias)
        

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return ModulatedDeformConv2dFunction._hcu_backward(
            ctx, grad_output)


modulated_deform_conv2d = ModulatedDeformConv2dFunction.apply
