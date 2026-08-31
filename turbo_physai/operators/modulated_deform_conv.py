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
from typing import Optional

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import hipdnn


_GRAPH_CACHE = {}
_UNSUPPORTED_FP16_BACKWARD = set()
_GRADIENT_NAMES = ("input", "offset", "mask", "weight", "bias")


def _hipdnn_dtype(dtype):
    if dtype == torch.float32:
        return hipdnn.data_type.FLOAT
    if dtype == torch.float16:
        return hipdnn.data_type.HALF
    raise ValueError(f"hipDNN deformable convolution does not support {dtype}")


def _tensor_signature(tensor):
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        tensor.device.index,
    )


def _graph_key(kind, tensors, stride, padding, dilation):
    return (
        kind,
        tuple(_tensor_signature(tensor) for tensor in tensors),
        tuple(stride),
        tuple(padding),
        tuple(dilation),
    )


def build_fprop_graph(input, offset, weight, mask, stride, padding, dilation):
    hipdnn_dtype = _hipdnn_dtype(input.dtype)

    graph = hipdnn.pygraph(
        name="deform_convolution",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn.data_type.FLOAT,
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

    return graph, input_h, offset_h, weight_h, mask_h, out


def build_wrw_graph(
    input, offset, weight, mask, grad_output, stride, padding, dilation
):
    hipdnn_dtype = _hipdnn_dtype(input.dtype)

    graph = hipdnn.pygraph(
        name="deform_convolution_wrw",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn.data_type.FLOAT,
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


def build_dx_graph(
    input,
    offset,
    weight,
    mask,
    grad_output,
    stride,
    padding,
    dilation,
    required_gradients,
):
    hipdnn_dtype = _hipdnn_dtype(input.dtype)

    graph = hipdnn.pygraph(
        name="deform_convolution_bwd",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn.data_type.FLOAT,
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
    need_input, need_offset, need_mask = required_gradients
    if need_input:
        dx.set_dim(input.shape).set_output(True).set_data_type(hipdnn_dtype)
    if need_offset:
        doffset.set_dim(offset.shape).set_output(True).set_data_type(hipdnn_dtype)
    if need_mask:
        dmask.set_dim(mask.shape).set_output(True).set_data_type(hipdnn_dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()

    return (
        graph,
        input_h,
        offset_h,
        weight_h,
        mask_h,
        grad_output_h,
        dx,
        doffset,
        dmask,
    )


class ModulatedDeformConv2dFunction(Function):
    @staticmethod
    def _output_size(ctx, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0),)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = ctx.padding[d]
            kernel = ctx.dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = ctx.stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1,)
        output_size = (output_size[0], channels, *output_size[1:])
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                "convolution input is too small (output would be "
                + "x".join(map(str, output_size))
                + ")"
            )
        return output_size

    @staticmethod
    def _memory_format(input):
        return (
            torch.channels_last
            if input.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )

    @staticmethod
    def _hcu_forward_single(ctx, input, offset, mask, weight):
        memory_format = (
            torch.channels_last
            if input.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        output = torch.empty(
            ModulatedDeformConv2dFunction._output_size(ctx, input, weight),
            dtype=input.dtype,
            device=input.device,
            memory_format=memory_format,
        )
        shape_key = _graph_key(
            "fprop",
            (input, offset, weight, mask),
            ctx.stride,
            ctx.padding,
            ctx.dilation,
        )
        if shape_key not in _GRAPH_CACHE:
            graph_data = build_fprop_graph(
                input, offset, weight, mask, ctx.stride, ctx.padding, ctx.dilation
            )
            _GRAPH_CACHE[shape_key] = graph_data

        (graph, input_h, offset_h, weight_h, mask_h, out) = _GRAPH_CACHE[shape_key]

        variant_pack = {
            input_h: input.data_ptr(),
            offset_h: offset.data_ptr(),
            weight_h: weight.data_ptr(),
            mask_h: mask.data_ptr(),
            out: output.data_ptr(),
        }
        workspace = torch.zeros(
            graph.get_workspace_size(), dtype=torch.uint8, device=input.device
        )

        graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())
        return output

    @staticmethod
    def _hcu_forward(ctx, input, offset, mask, weight, bias):
        if ctx.deform_groups == 1:
            output = ModulatedDeformConv2dFunction._hcu_forward_single(
                ctx, input, offset, mask, weight
            )
        else:
            memory_format = ModulatedDeformConv2dFunction._memory_format(input)
            channels_per_group = input.shape[1] // ctx.deform_groups
            offset_channels = offset.shape[1] // ctx.deform_groups
            mask_channels = mask.shape[1] // ctx.deform_groups
            output = None
            for group_index in range(ctx.deform_groups):
                input_slice = slice(
                    group_index * channels_per_group,
                    (group_index + 1) * channels_per_group,
                )
                offset_slice = slice(
                    group_index * offset_channels,
                    (group_index + 1) * offset_channels,
                )
                mask_slice = slice(
                    group_index * mask_channels,
                    (group_index + 1) * mask_channels,
                )
                group_output = ModulatedDeformConv2dFunction._hcu_forward_single(
                    ctx,
                    input[:, input_slice].contiguous(memory_format=memory_format),
                    offset[:, offset_slice].contiguous(memory_format=memory_format),
                    mask[:, mask_slice].contiguous(memory_format=memory_format),
                    weight[:, input_slice].contiguous(memory_format=memory_format),
                )
                output = group_output if output is None else output + group_output
        if ctx.with_bias:
            output = output + bias.reshape(1, -1, 1, 1)
        return output

    @staticmethod
    def _hcu_backward_single_impl(
        ctx,
        input,
        offset,
        mask,
        weight,
        grad_output,
        required_gradients,
    ):
        need_input, need_offset, need_mask, need_weight = required_gradients
        grad_input = torch.zeros_like(input) if need_input else None
        grad_offset = torch.zeros_like(offset) if need_offset else None
        grad_mask = torch.zeros_like(mask) if need_mask else None
        grad_weight = torch.zeros_like(weight) if need_weight else None

        if need_weight:
            shape_key = _graph_key(
                "wgrad",
                (input, offset, weight, mask, grad_output),
                ctx.stride,
                ctx.padding,
                ctx.dilation,
            )
            if shape_key not in _GRAPH_CACHE:
                graph_data = build_wrw_graph(
                    input,
                    offset,
                    weight,
                    mask,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                )
                _GRAPH_CACHE[shape_key] = graph_data

            (graph, input_h, offset_h, mask_h, grad_output_h, dw) = _GRAPH_CACHE[
                shape_key
            ]
            variant_pack = {
                input_h: input.data_ptr(),
                offset_h: offset.data_ptr(),
                grad_output_h: grad_output.data_ptr(),
                mask_h: mask.data_ptr(),
                dw: grad_weight.data_ptr(),
            }
            workspace = torch.zeros(
                graph.get_workspace_size(),
                dtype=torch.uint8,
                device=input.device,
            )
            graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())

        dgrad_selection = (need_input, need_offset, need_mask)
        if any(dgrad_selection):
            selection_key = "".join("1" if value else "0" for value in dgrad_selection)
            shape_key = _graph_key(
                f"dgrad:{selection_key}",
                (input, offset, weight, mask, grad_output),
                ctx.stride,
                ctx.padding,
                ctx.dilation,
            )
            if shape_key not in _GRAPH_CACHE:
                graph_data = build_dx_graph(
                    input,
                    offset,
                    weight,
                    mask,
                    grad_output,
                    ctx.stride,
                    ctx.padding,
                    ctx.dilation,
                    dgrad_selection,
                )
                _GRAPH_CACHE[shape_key] = graph_data

            (
                graph,
                input_h,
                offset_h,
                weight_h,
                mask_h,
                grad_output_h,
                dx,
                doffset,
                dmask,
            ) = _GRAPH_CACHE[shape_key]
            variant_pack = {
                grad_output_h: grad_output.data_ptr(),
                weight_h: weight.data_ptr(),
                offset_h: offset.data_ptr(),
                mask_h: mask.data_ptr(),
            }
            if need_offset or need_mask:
                variant_pack[input_h] = input.data_ptr()
            if need_input:
                variant_pack[dx] = grad_input.data_ptr()
            if need_offset:
                variant_pack[doffset] = grad_offset.data_ptr()
            if need_mask:
                variant_pack[dmask] = grad_mask.data_ptr()
            workspace = torch.zeros(
                graph.get_workspace_size(),
                dtype=torch.uint8,
                device=input.device,
            )
            graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())

        return grad_input, grad_offset, grad_mask, grad_weight

    @staticmethod
    def _hcu_backward_single(
        ctx,
        input,
        offset,
        mask,
        weight,
        grad_output,
        required_gradients,
    ):
        if input.dtype != torch.float16:
            return ModulatedDeformConv2dFunction._hcu_backward_single_impl(
                ctx,
                input,
                offset,
                mask,
                weight,
                grad_output,
                required_gradients,
            )

        selection_key = "".join("1" if value else "0" for value in required_gradients)
        support_key = _graph_key(
            f"fp16_backward:{selection_key}",
            (input, offset, weight, mask, grad_output),
            ctx.stride,
            ctx.padding,
            ctx.dilation,
        )
        if support_key not in _UNSUPPORTED_FP16_BACKWARD:
            try:
                return ModulatedDeformConv2dFunction._hcu_backward_single_impl(
                    ctx,
                    input,
                    offset,
                    mask,
                    weight,
                    grad_output,
                    required_gradients,
                )
            except RuntimeError as error:
                if "No engine configurations available" not in str(error):
                    raise
                _UNSUPPORTED_FP16_BACKWARD.add(support_key)

        memory_format = ModulatedDeformConv2dFunction._memory_format(input)
        fp32_gradients = ModulatedDeformConv2dFunction._hcu_backward_single_impl(
            ctx,
            input.float().contiguous(memory_format=memory_format),
            offset.float().contiguous(memory_format=memory_format),
            mask.float().contiguous(memory_format=memory_format),
            weight.float().contiguous(memory_format=memory_format),
            grad_output.float().contiguous(memory_format=memory_format),
            required_gradients,
        )
        return tuple(
            gradient.to(dtype=input.dtype) if gradient is not None else None
            for gradient in fp32_gradients
        )

    @staticmethod
    def _hcu_backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        autograd_needs = ctx.needs_input_grad[:5]
        required_gradients = ctx.required_gradients
        if required_gradients is None:
            selected_gradients = autograd_needs
        else:
            selected_gradients = tuple(
                autograd_needs[index] and name in required_gradients
                for index, name in enumerate(_GRADIENT_NAMES)
            )
        need_input, need_offset, need_mask, need_weight, need_bias = selected_gradients
        grad_bias = (
            grad_output.sum(dim=(0, 2, 3)) if ctx.with_bias and need_bias else None
        )
        memory_format = ModulatedDeformConv2dFunction._memory_format(input)
        grad_output = grad_output.contiguous(memory_format=memory_format)
        tensor_gradients = (
            need_input,
            need_offset,
            need_mask,
            need_weight,
        )

        if ctx.deform_groups == 1:
            gradients = ModulatedDeformConv2dFunction._hcu_backward_single(
                ctx,
                input,
                offset,
                mask,
                weight,
                grad_output,
                tensor_gradients,
            )
            grad_input, grad_offset, grad_mask, grad_weight = gradients
        else:
            grad_input = torch.zeros_like(input) if need_input else None
            grad_offset = torch.zeros_like(offset) if need_offset else None
            grad_mask = torch.zeros_like(mask) if need_mask else None
            grad_weight = torch.zeros_like(weight) if need_weight else None
            channels_per_group = input.shape[1] // ctx.deform_groups
            offset_channels = offset.shape[1] // ctx.deform_groups
            mask_channels = mask.shape[1] // ctx.deform_groups
            for group_index in range(ctx.deform_groups):
                input_slice = slice(
                    group_index * channels_per_group,
                    (group_index + 1) * channels_per_group,
                )
                offset_slice = slice(
                    group_index * offset_channels,
                    (group_index + 1) * offset_channels,
                )
                mask_slice = slice(
                    group_index * mask_channels,
                    (group_index + 1) * mask_channels,
                )
                group_gradients = ModulatedDeformConv2dFunction._hcu_backward_single(
                    ctx,
                    input[:, input_slice].contiguous(memory_format=memory_format),
                    offset[:, offset_slice].contiguous(memory_format=memory_format),
                    mask[:, mask_slice].contiguous(memory_format=memory_format),
                    weight[:, input_slice].contiguous(memory_format=memory_format),
                    grad_output,
                    tensor_gradients,
                )
                (
                    group_grad_input,
                    group_grad_offset,
                    group_grad_mask,
                    group_grad_weight,
                ) = group_gradients
                if need_input:
                    grad_input[:, input_slice] = group_grad_input
                if need_offset:
                    grad_offset[:, offset_slice] = group_grad_offset
                if need_mask:
                    grad_mask[:, mask_slice] = group_grad_mask
                if need_weight:
                    grad_weight[:, input_slice] = group_grad_weight

        return (
            grad_input,
            grad_offset,
            grad_mask,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1,
        required_gradients=None,
    ) -> torch.Tensor:
        if input is not None and input.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor as input, got {input.dim()}D tensor \
                  instead."
            )
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deform_groups = deform_groups
        if required_gradients is None:
            ctx.required_gradients = None
        else:
            required_gradients = frozenset(required_gradients)
            unknown_gradients = required_gradients.difference(_GRADIENT_NAMES)
            if unknown_gradients:
                unknown = ", ".join(sorted(unknown_gradients))
                raise ValueError(f"unknown required gradients: {unknown}")
            ctx.required_gradients = required_gradients
        if groups != 1:
            raise ValueError(
                "hipDNN deformable convolution does not support groups != 1"
            )
        if not isinstance(deform_groups, int) or deform_groups < 1:
            raise ValueError("deform_groups must be a positive integer")
        kernel_points = weight.shape[-2] * weight.shape[-1]
        if input.shape[1] % deform_groups != 0:
            raise ValueError("input channels must be divisible by deform_groups")
        if offset.shape[1] != 2 * deform_groups * kernel_points:
            raise ValueError(
                "offset channels do not match deform_groups and kernel size"
            )
        if mask.shape[1] != deform_groups * kernel_points:
            raise ValueError("mask channels do not match deform_groups and kernel size")
        if bias is not None and bias.numel() != weight.shape[0]:
            raise ValueError("bias size must match output channels")
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
        mask = mask.type_as(input)
        bias = bias.type_as(input)  # type: ignore
        memory_format = (
            torch.channels_last
            if input.is_contiguous(memory_format=torch.channels_last)
            else torch.contiguous_format
        )
        input = input.contiguous(memory_format=memory_format)
        offset = offset.contiguous(memory_format=memory_format)
        mask = mask.contiguous(memory_format=memory_format)
        weight = weight.contiguous(memory_format=memory_format)
        bias = bias.contiguous()
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return ModulatedDeformConv2dFunction._hcu_forward(
            ctx, input, offset, mask, weight, bias
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return ModulatedDeformConv2dFunction._hcu_backward(ctx, grad_output)


def modulated_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    deform_groups: int = 1,
    *,
    required_gradients=None,
) -> torch.Tensor:
    """Apply MDC and optionally restrict its backward outputs.

    ``required_gradients=None`` follows PyTorch ``requires_grad``. Model-side
    adapters may request a subset of ``input``, ``offset``, ``mask``,
    ``weight`` and ``bias`` without changing MMCV's positional contract.
    """

    return ModulatedDeformConv2dFunction.apply(
        input,
        offset,
        mask,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        deform_groups,
        required_gradients,
    )
