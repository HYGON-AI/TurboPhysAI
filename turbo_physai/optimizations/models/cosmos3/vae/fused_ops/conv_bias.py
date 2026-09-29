# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional Conv3D + bias graph for a cache-free Wan VAE chunk."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from turbo_physai.optimizations.models.cosmos3.vae.fused_ops._hipdnn import (
    GraphEntry,
    conv_channels_supported,
    conv_fprop,
    empty_sentinel,
    execute_graph,
    fused_dtype_supported,
    fused_ops_enabled,
    get_custom_op_decorator,
    get_or_build_graph,
    graph_cache_key,
    hipdnn_data_type,
    hipdnn_handle,
    is_empty_sentinel,
    make_graph_entry,
    require_hipdnn,
    tensor_signature,
    to_canonical_ndhwc,
)

_GRAPH_CACHE: dict[tuple[Any, ...], GraphEntry | None] = {}
custom_op = get_custom_op_decorator()


def _bias_view(bias: torch.Tensor, out_channels: int) -> torch.Tensor:
    if bias.numel() != out_channels:
        raise ValueError(f"Conv bias must contain {out_channels} values, got {bias.shape}")
    return bias.reshape(1, out_channels, 1, 1, 1)


def _conv_output_shape(
    x: torch.Tensor,
    weight: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[int, ...]:
    output_dims = []
    for input_dim, kernel_dim, pre, post, step, dil in zip(
        x.shape[2:],
        weight.shape[2:],
        pre_padding,
        post_padding,
        stride,
        dilation,
    ):
        output_dims.append((input_dim + pre + post - dil * (kernel_dim - 1) - 1) // step + 1)
    return (x.shape[0], weight.shape[0], *output_dims)


def _validate_contract(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> None:
    if x.dim() != 5:
        raise ValueError(f"Fused Conv3D+bias expects a 5D input, got {x.shape}")
    triples = {
        "pre_padding": pre_padding,
        "post_padding": post_padding,
        "stride": stride,
        "dilation": dilation,
    }
    for name, values in triples.items():
        if len(values) != 3:
            raise ValueError(f"{name} must contain T/H/W values, got {values}")
    if any(value < 0 for value in (*pre_padding, *post_padding)):
        raise ValueError(f"Conv3D padding must be non-negative, got {pre_padding=} {post_padding=}")
    if any(value <= 0 for value in (*stride, *dilation)):
        raise ValueError(f"Conv3D stride/dilation must be positive, got {stride=} {dilation=}")
    if groups != 1:
        raise ValueError(f"Wan VAE fused Conv3D+bias requires groups=1, got {groups}")
    if weight.dim() != 5 or tuple(weight.shape[2:]) != (3, 3, 3):
        raise ValueError(f"Wan VAE fused Conv3D+bias requires [K,C,3,3,3], got {weight.shape}")
    if int(weight.shape[1]) != int(x.shape[1]):
        raise ValueError(f"Weight/input channel mismatch: {weight.shape=} {x.shape=}")
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("Conv3D input and weight must have the same dtype and device")
    if not is_empty_sentinel(bias):
        _bias_view(bias, int(weight.shape[0]))
        if bias.dtype != x.dtype or bias.device != x.device:
            raise ValueError("Conv3D bias and input must have the same dtype and device")


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> torch.Tensor:
    pad = (
        pre_padding[2],
        post_padding[2],
        pre_padding[1],
        post_padding[1],
        pre_padding[0],
        post_padding[0],
    )
    image = F.pad(x, pad)
    conv_bias = None if is_empty_sentinel(bias) else bias.reshape(-1)
    output = F.conv3d(
        image,
        weight,
        conv_bias,
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=groups,
    )
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        output = output.contiguous(memory_format=torch.channels_last_3d)
    return output


def _build_graph(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> GraphEntry:
    hipdnn = require_hipdnn()
    graph = hipdnn.pygraph(
        handle=hipdnn_handle(hipdnn, x.device),
        io_data_type=hipdnn_data_type(hipdnn, x.dtype),
        intermediate_data_type=hipdnn.data_type.FLOAT,
        compute_data_type=hipdnn.data_type.FLOAT,
        name="wan_vae_conv_bias",
    )
    x_tensor = graph.tensor_like(x)
    weight_tensor = graph.tensor_like(weight)
    inputs: dict[str, Any] = {"x": x_tensor, "weight": weight_tensor}
    output_tensor = conv_fprop(
        graph,
        x_tensor,
        weight_tensor,
        pre_padding,
        post_padding,
        stride,
        dilation,
        "conv3d",
    )
    if not is_empty_sentinel(bias):
        bias_tensor = graph.tensor_like(bias)
        inputs["bias"] = bias_tensor
        output_tensor = graph.add(a=output_tensor, b=bias_tensor, name="bias")
    output_tensor.set_output(True)
    return make_graph_entry(graph, inputs, output_tensor, x)


@custom_op("turbo_physai::wan_vae_conv_bias", mutates_args=())
def _conv_bias_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_t: int,
    pre_h: int,
    pre_w: int,
    post_t: int,
    post_h: int,
    post_w: int,
    stride_t: int,
    stride_h: int,
    stride_w: int,
    dilation_t: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> torch.Tensor:
    pre_padding = (pre_t, pre_h, pre_w)
    post_padding = (post_t, post_h, post_w)
    stride = (stride_t, stride_h, stride_w)
    dilation = (dilation_t, dilation_h, dilation_w)
    key = graph_cache_key(
        tensor_signature(x),
        tensor_signature(weight),
        tensor_signature(bias),
        pre_padding,
        post_padding,
        stride,
        dilation,
        int(groups),
    )
    entry = get_or_build_graph(
        _GRAPH_CACHE,
        key,
        lambda: _build_graph(x, weight, bias, pre_padding, post_padding, stride, dilation),
        what="Conv3D+bias",
        describe=lambda: (
            f"x={tensor_signature(x)}, weight={tensor_signature(weight)}, "
            f"bias={tensor_signature(bias)}, pre_padding={pre_padding}, "
            f"post_padding={post_padding}, stride={stride}, dilation={dilation}, "
            f"groups={groups}"
        ),
    )
    if entry is None:
        return _reference(x, weight, bias, pre_padding, post_padding, stride, dilation, int(groups))
    bindings: dict[str, torch.Tensor] = {"x": x, "weight": weight}
    if not is_empty_sentinel(bias):
        bindings["bias"] = bias
    return execute_graph(entry, bindings, x)


@_conv_bias_op.register_fake
def _conv_bias_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_t: int,
    pre_h: int,
    pre_w: int,
    post_t: int,
    post_h: int,
    post_w: int,
    stride_t: int,
    stride_h: int,
    stride_w: int,
    dilation_t: int,
    dilation_h: int,
    dilation_w: int,
    groups: int,
) -> torch.Tensor:
    del bias, groups
    shape = _conv_output_shape(
        x,
        weight,
        (pre_t, pre_h, pre_w),
        (post_t, post_h, post_w),
        (stride_t, stride_h, stride_w),
        (dilation_t, dilation_h, dilation_w),
    )
    return torch.empty(
        shape,
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )


def fused_conv3d_bias_ndhwc(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    pre_padding: tuple[int, int, int] = (2, 1, 1),
    post_padding: tuple[int, int, int] = (0, 1, 1),
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Run exactly Conv3D + optional bias for a cache-free causal chunk."""

    bias_view = empty_sentinel(x) if bias is None else _bias_view(bias, int(weight.shape[0]))
    _validate_contract(x, weight, bias_view, pre_padding, post_padding, stride, dilation, groups)
    eligible = (
        fused_ops_enabled()
        and fused_dtype_supported(x.dtype)
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and weight.is_contiguous(memory_format=torch.channels_last_3d)
        and conv_channels_supported(int(x.shape[1]), int(weight.shape[0]))
        and not any(tensor.requires_grad for tensor in (x, weight, bias_view))
    )
    if not eligible:
        return _reference(x, weight, bias_view, pre_padding, post_padding, stride, dilation, groups)

    x = to_canonical_ndhwc(x)
    weight = to_canonical_ndhwc(weight)
    return _conv_bias_op(
        x,
        weight,
        bias_view,
        *pre_padding,
        *post_padding,
        *stride,
        *dilation,
        int(groups),
    )


__all__ = ["fused_conv3d_bias_ndhwc"]
