# Copyright 2018-2019 OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
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
# Modified by Hygon.

"""BEVFormer-specific hipDNN MDC implementation and compile adapter."""

import math
from typing import Tuple, Union
import os
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _single
import hipdnn
from mmcv.utils import deprecated_api_warning
from mmcv.cnn import CONV_LAYERS
from mmcv.utils import ext_loader, print_log

ext_module = ext_loader.load_ext(
    '_ext',
    ['modulated_deform_conv_forward', 'modulated_deform_conv_backward'])


shape_dict = {}


def _param_key(values):
    return 'x'.join(str(int(value)) for value in values)


def _conv_key(kind, input, weight, offset, mask, stride, padding, dilation,
              grad_output=None):
    shape_parts = []
    for t in (input, weight, offset, mask):
        shape_parts.append('x'.join(str(int(dim)) for dim in t.shape))
    if grad_output is not None:
        shape_parts.append('x'.join(str(int(dim)) for dim in grad_output.shape))
    shape_key = '|'.join(shape_parts)

    return ':'.join((
        kind,
        shape_key,
        str(input.dtype),
        str(input.device),
        's' + _param_key(stride),
        'p' + _param_key(padding),
        'd' + _param_key(dilation),
    ))


def _debug_cache_build(kind, shape_key):
    if os.getenv('BEVFORMER_DCN_DEBUG', '0') == '1':
        print(f'[DCN cache build] pid={os.getpid()} kind={kind} '
              f'cache_size={len(shape_dict)} key={shape_key}', flush=True)


def _output_size(input, weight, stride, padding, dilation, check=True):
    channels = weight.size(0)
    output_size = (input.size(0), channels)
    for d in range(input.dim() - 2):
        in_size = input.size(d + 2)
        pad = padding[d]
        kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
        stride_ = stride[d]
        output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
    if check and not all(map(lambda s: s > 0, output_size)):
        raise ValueError('convolution input is too small (output would be ' +
                         'x'.join(map(str, output_size)) + ')')
    return output_size


def _miopen_output_size(input, weight, stride, padding, dilation):
    channels = weight.size(0)
    output_size = (input.size(0),)
    for d in range(input.dim() - 2):
        in_size = input.size(d + 2)
        pad = padding[d]
        kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
        stride_ = stride[d]
        output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
    output_size += (channels,)
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError('convolution input is too small (output would be ' +
                         'x'.join(map(str, output_size)) + ')')
    return output_size


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
        #dk_test
        compute_data_type=hipdnn.data_type.FLOAT,
        #compute_data_type=hipdnn_dtype,
    )

    input_h = graph.tensor_like(input.detach())
    offset_h = graph.tensor_like(offset.detach())
    weight_h = graph.tensor_like(weight.detach())
    mask_h = graph.tensor_like(mask.detach())

    out = graph.deform_conv_fprop(
        # dk_test
        image=input_h,
        #input=input_h,
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
        #dk_test
        compute_data_type=hipdnn.data_type.FLOAT,
        #compute_data_type=hipdnn_dtype,
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

# dk_test
'''
def build_fusion_graph(input, offset, weight, mask, grad_output, stride, padding, dilation):
    hipdnn_dtype = hipdnn.data_type.FLOAT
    if input.dtype == torch.float32:
        hipdnn_dtype = hipdnn.data_type.FLOAT
    elif input.dtype == torch.float16:
        hipdnn_dtype = hipdnn.data_type.HALF
    else:
        raise ValueError(
                f"miopen deformer conv not support dtyp {input.dtype}")

    graph = hipdnn.pygraph(
        name="deformable_convolution_bwd_fusion",
        io_data_type=hipdnn_dtype,
        intermediate_data_type=hipdnn_dtype,
        compute_data_type=hipdnn_dtype,
    )

    input_h = graph.tensor_like(input.detach())
    offset_h = graph.tensor_like(offset.detach())
    weight_h = graph.tensor_like(weight.detach())
    mask_h = graph.tensor_like(mask.detach())
    grad_output_h = graph.tensor_like(grad_output.detach())

    dx, doffset, dmask = graph.deform_conv_fusiongrad(
        loss=grad_output_h,
        image=input_h,
        offset=offset_h,
        weight=weight_h,
        mask=mask_h,
        stride=stride,
        padding=padding,
        dilation=dilation,
        name="deform_conv_bwd_fusion",
    )
    dx.set_dim(input.shape).set_output(True).set_data_type(hipdnn_dtype)
    doffset.set_dim(offset.shape).set_output(True).set_data_type(hipdnn_dtype)
    dmask.set_dim(mask.shape).set_output(True).set_data_type(hipdnn_dtype)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans()
    graph.check_support()
    graph.build_plans()

    return (graph, input_h, offset_h, weight_h, mask_h, grad_output_h, dx, doffset, dmask)
'''


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
        #dk_test
        compute_data_type=hipdnn.data_type.FLOAT,
        #compute_data_type=hipdnn_dtype,
    )

    offset_h = graph.tensor_like(offset.detach())
    weight_h = graph.tensor_like(weight.detach())
    mask_h = graph.tensor_like(mask.detach())
    grad_output_h = graph.tensor_like(grad_output.detach())
    input_h = graph.tensor_like(input.detach())

    #dk_test
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

    return (graph, offset_h, weight_h, mask_h, grad_output_h, dx)


def _hcu_forward(input, offset, mask, weight, stride, padding, dilation):
    fp16_mode = os.getenv('ENABLE_MIOPEN_DEFORMER_CONV_FP16',
                          'False').lower() == 'true'
    origin_type = input.dtype
    if fp16_mode and origin_type == torch.float32:
        input = input.to(dtype=torch.float16)
        offset = offset.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
        weight = weight.to(dtype=torch.float16)

    output = input.new_empty([
        int(i) for i in _miopen_output_size(input, weight, stride, padding,
                                            dilation)
    ])
    shape_key = _conv_key('fwd', input, weight, offset, mask, stride, padding,
                          dilation)
    if shape_key not in shape_dict:
        _debug_cache_build('fwd', shape_key)
        graph, input_h, offset_h, weight_h, mask_h, out = build_fprop_graph(
            input, offset, weight, mask, stride, padding, dilation)
        workspace = torch.empty(
            graph.get_workspace_size(), dtype=torch.uint8, device=input.device)
        shape_dict[shape_key] = [
            graph, input_h, offset_h, weight_h, mask_h, out, workspace
        ]

    graph, input_h, offset_h, weight_h, mask_h, out, workspace = shape_dict[
        shape_key]
    variant_pack = {
        input_h: input.data_ptr(),
        offset_h: offset.data_ptr(),
        weight_h: weight.data_ptr(),
        mask_h: mask.data_ptr(),
        out: output.data_ptr(),
    }
    graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())
    if fp16_mode and origin_type == torch.float32:
        output = output.to(dtype=torch.float32)

    return output.permute(0, 3, 1, 2)


def _hcu_backward(input, offset, mask, weight, grad_output, stride, padding,
                  dilation):
    fp16_mode = os.getenv('ENABLE_MIOPEN_DEFORMER_CONV_FP16',
                          'False').lower() == 'true'
    origin_type = input.dtype
    if fp16_mode and origin_type == torch.float32:
        input = input.to(dtype=torch.float16)
        offset = offset.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
        weight = weight.to(dtype=torch.float16)
        grad_output = grad_output.to(dtype=torch.float16)

    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)
    grad_output = grad_output.to(memory_format=torch.channels_last)

    shape_key = _conv_key('wrw', input, weight, offset, mask, stride, padding,
                          dilation, grad_output)
    if shape_key not in shape_dict:
        _debug_cache_build('wrw', shape_key)
        graph, input_h, offset_h, mask_h, grad_output_h, dw = build_wrw_graph(
            input, offset, weight, mask, grad_output, stride, padding,
            dilation)
        workspace = torch.empty(
            graph.get_workspace_size(), dtype=torch.uint8, device=input.device)
        shape_dict[shape_key] = [
            graph, input_h, offset_h, mask_h, grad_output_h, dw, workspace
        ]

    graph, input_h, offset_h, mask_h, grad_output_h, dw, workspace = shape_dict[
        shape_key]
    variant_pack = {
        input_h: input.data_ptr(),
        offset_h: offset.data_ptr(),
        grad_output_h: grad_output.data_ptr(),
        mask_h: mask.data_ptr(),
        dw: grad_weight.data_ptr(),
    }
    graph.exec(variant_pack=variant_pack, workspace=workspace.data_ptr())

    shape_key = _conv_key('dx', input, weight, offset, mask, stride, padding,
                          dilation, grad_output)
    if shape_key not in shape_dict:
        _debug_cache_build('dx', shape_key)
        graph, offset_h, weight_h, mask_h, grad_output_h, dx = build_dx_graph(
            input, offset, weight, mask, grad_output, stride, padding,
            dilation)
        workspace_dx = torch.empty(
            graph.get_workspace_size(), dtype=torch.uint8, device=input.device)
        shape_dict[shape_key] = [
            graph, offset_h, weight_h, mask_h, grad_output_h, dx, workspace_dx
        ]

    (graph, offset_h, weight_h, mask_h, grad_output_h, dx,
     workspace_dx) = shape_dict[shape_key]
    variant_pack = {
        grad_output_h: grad_output.data_ptr(),
        weight_h: weight.data_ptr(),
        offset_h: offset.data_ptr(),
        mask_h: mask.data_ptr(),
        dx: grad_input.data_ptr(),
    }
    graph.exec(variant_pack=variant_pack, workspace=workspace_dx.data_ptr())

    if fp16_mode and origin_type == torch.float32:
        grad_input = grad_input.to(dtype=torch.float32)
        grad_weight = grad_weight.to(dtype=torch.float32)

    return grad_input, grad_weight


def _mmcv_forward(input, offset, mask, weight, bias, stride, padding,
                  dilation, groups, deform_groups, with_bias):
    bufs = [input.new_empty(0), input.new_empty(0)]
    output = input.new_empty(_output_size(input, weight, stride, padding,
                                          dilation))
    ext_module.modulated_deform_conv_forward(
        input.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        bufs[0],
        offset.contiguous(),
        mask.contiguous(),
        output,
        bufs[1],
        kernel_h=weight.size(2),
        kernel_w=weight.size(3),
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        group=groups,
        deformable_group=deform_groups,
        with_bias=with_bias)
    return output


def _mmcv_backward(input, offset, mask, weight, bias, grad_output, stride,
                   padding, dilation, groups, deform_groups, with_bias):
    input = input.contiguous()
    offset = offset.contiguous()
    mask = mask.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    bufs = [input.new_empty(0), input.new_empty(0)]

    grad_input = torch.zeros_like(input)
    grad_offset = torch.zeros_like(offset)
    grad_mask = torch.zeros_like(mask)
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias)
    grad_output = grad_output.contiguous()
    ext_module.modulated_deform_conv_backward(
        input,
        weight,
        bias,
        bufs[0],
        offset,
        mask,
        bufs[1],
        grad_input,
        grad_weight,
        grad_bias,
        grad_offset,
        grad_mask,
        grad_output,
        kernel_h=weight.size(2),
        kernel_w=weight.size(3),
        stride_h=stride[0],
        stride_w=stride[1],
        pad_h=padding[0],
        pad_w=padding[1],
        dilation_h=dilation[0],
        dilation_w=dilation[1],
        group=groups,
        deformable_group=deform_groups,
        with_bias=with_bias)
    return grad_input, grad_offset, grad_mask, grad_weight, grad_bias


@torch.library.custom_op("mmcv::modulated_deform_conv2d", mutates_args=())
def _modulated_deform_conv2d(input: torch.Tensor, offset: torch.Tensor,
                             mask: torch.Tensor, weight: torch.Tensor,
                             bias: torch.Tensor, stride_h: int,
                             stride_w: int, pad_h: int, pad_w: int,
                             dilation_h: int, dilation_w: int, groups: int,
                             deform_groups: int,
                             with_bias: bool) -> torch.Tensor:
    if input is not None and input.dim() != 4:
        raise ValueError(
            f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')

    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    dilation = (dilation_h, dilation_w)
    if with_bias:
        raise NotImplementedError(
            "TurboPhysAI hipDNN ModulatedDeformConv2d does not support bias"
        )
    return _hcu_forward(input, offset, mask, weight, stride, padding,
                        dilation)


@_modulated_deform_conv2d.register_fake
def _(input: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor,
      weight: torch.Tensor, bias: torch.Tensor, stride_h: int, stride_w: int,
      pad_h: int, pad_w: int, dilation_h: int, dilation_w: int, groups: int,
      deform_groups: int, with_bias: bool) -> torch.Tensor:
    if not with_bias:
        output_size = _output_size(input, weight, (stride_h, stride_w),
                                   (pad_h, pad_w),
                                   (dilation_h, dilation_w), check=False)
        output_nhwc = input.new_empty(
            (output_size[0], output_size[2], output_size[3], output_size[1]))
        return output_nhwc.permute(0, 3, 1, 2)

    return input.new_empty(
        _output_size(input, weight, (stride_h, stride_w), (pad_h, pad_w),
                     (dilation_h, dilation_w), check=False))


@torch.library.custom_op("mmcv::modulated_deform_conv2d_backward",
                         mutates_args=())
def _modulated_deform_conv2d_backward(
        input: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor,
        weight: torch.Tensor, bias: torch.Tensor,
        grad_output: torch.Tensor, stride_h: int, stride_w: int, pad_h: int,
        pad_w: int, dilation_h: int, dilation_w: int, groups: int,
        deform_groups: int,
        with_bias: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                  torch.Tensor, torch.Tensor]:
    stride = (stride_h, stride_w)
    padding = (pad_h, pad_w)
    dilation = (dilation_h, dilation_w)

    if with_bias:
        raise NotImplementedError(
            "TurboPhysAI hipDNN ModulatedDeformConv2d does not support bias"
        )

    grad_input, grad_weight = _hcu_backward(
        input, offset, mask, weight, grad_output, stride, padding, dilation)
    return (grad_input, offset.new_empty(0), mask.new_empty(0), grad_weight,
            bias.new_empty(0))


@_modulated_deform_conv2d_backward.register_fake
def _(input: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor,
      weight: torch.Tensor, bias: torch.Tensor, grad_output: torch.Tensor,
      stride_h: int, stride_w: int, pad_h: int, pad_w: int, dilation_h: int,
      dilation_w: int, groups: int, deform_groups: int,
      with_bias: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor]:
    if with_bias:
        return (torch.empty_like(input), torch.empty_like(offset),
                torch.empty_like(mask), torch.empty_like(weight),
                torch.empty_like(bias))
    return (torch.empty_like(input), offset.new_empty(0), mask.new_empty(0),
            torch.empty_like(weight), bias.new_empty(0))


def _modulated_deform_conv2d_setup_context(ctx, inputs, output):
    (input, offset, mask, weight, bias, stride_h, stride_w, pad_h, pad_w,
     dilation_h, dilation_w, groups, deform_groups, with_bias) = inputs
    ctx.save_for_backward(input, offset, mask, weight, bias)
    ctx.stride_h = stride_h
    ctx.stride_w = stride_w
    ctx.pad_h = pad_h
    ctx.pad_w = pad_w
    ctx.dilation_h = dilation_h
    ctx.dilation_w = dilation_w
    ctx.groups = groups
    ctx.deform_groups = deform_groups
    ctx.with_bias = with_bias


def _modulated_deform_conv2d_autograd(ctx, grad_output):
    input, offset, mask, weight, bias = ctx.saved_tensors
    grads = _modulated_deform_conv2d_backward(
        input, offset, mask, weight, bias, grad_output, ctx.stride_h,
        ctx.stride_w, ctx.pad_h, ctx.pad_w, ctx.dilation_h, ctx.dilation_w,
        ctx.groups, ctx.deform_groups, ctx.with_bias)

    grad_input, grad_offset, grad_mask, grad_weight, grad_bias = grads
    if ctx.with_bias:
        return (grad_input, grad_offset, grad_mask, grad_weight, grad_bias,
                None, None, None, None, None, None, None, None, None)

    return (grad_input, None, None, grad_weight, None, None, None, None, None,
            None, None, None, None, None)


torch.library.register_autograd(
    "mmcv::modulated_deform_conv2d",
    _modulated_deform_conv2d_autograd,
    setup_context=_modulated_deform_conv2d_setup_context,
)


def _modulated_deform_conv2d_dispatch(
        input: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
        bias: Union[torch.Tensor, None] = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deform_groups: int = 1) -> torch.Tensor:
    with_bias = bias is not None
    if bias is None:
        bias = input.new_empty(0)

    input = input.type_as(offset)
    weight = weight.type_as(input)
    bias = bias.type_as(input)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    return _modulated_deform_conv2d(input, offset, mask, weight, bias,
                                    stride[0], stride[1], padding[0],
                                    padding[1], dilation[0], dilation[1],
                                    groups, deform_groups, with_bias)


def modulated_deform_conv2d(*args, **kwargs):
    """MMCV-compatible public entry; the baseline exposes a decorated variadic API."""

    return _modulated_deform_conv2d_dispatch(*args, **kwargs)


class ModulatedDeformConv2d(nn.Module):

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='ModulatedDeformConv2d')
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: Union[bool, str] = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups,
                         *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x: torch.Tensor, offset: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        return _modulated_deform_conv2d_dispatch(
            x, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deform_groups)


class ModulatedDeformConv2dPack(ModulatedDeformConv2d):
    """A ModulatedDeformable Conv Encapsulation that acts as normal Conv
    layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int): Same as nn.Conv2d, while tuple is not supported.
        padding (int): Same as nn.Conv2d, while tuple is not supported.
        dilation (int): Same as nn.Conv2d, while tuple is not supported.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True)
        self.init_weights()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        # kernel: kme_implicitgemm_fp16_fwd_64x64x16_common
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # dk_test
        # print("#"*100)
        # print("ModulatedDeformConv2dPack modulated_deform_conv2d info: ",x.shape,
        #          offset.shape,mask.shape,self.weight.shape,self.bias,
        #         self.stride,
        #         self.padding,
        #         self.dilation,
        #         self.groups,
        #         self.deform_groups)
        return _modulated_deform_conv2d_dispatch(
            x, offset, mask, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups, self.deform_groups)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, ModulatedDeformConvPack
            # loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'ModulatedDeformConvPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
