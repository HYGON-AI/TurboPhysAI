# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure Concat + causal Conv3d + bias graph for Wan VAE."""

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
    fused_concat_enabled,
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
from turbo_physai.optimizations.models.cosmos3.vae.fused_ops.conv_bias import (
    fused_conv3d_bias_ndhwc,
)

_GRAPH_CACHE: dict[tuple[Any, ...], GraphEntry | None] = {}
custom_op = get_custom_op_decorator()


def _new_cache(x: torch.Tensor) -> torch.Tensor:
    """Return exactly the newest min(2, T) frames required by requirement two.

    ``clone(memory_format=channels_last_3d)`` ensures that even if N=1 or T=1,
    the slice has normalized canonical strides.
    """

    return x[:, :, -2:, :, :].clone(memory_format=torch.channels_last_3d)


def _cache_frames(cache: torch.Tensor) -> int:
    """Temporal length of *cache*, with the empty sentinel meaning Prime."""

    return 0 if is_empty_sentinel(cache) else int(cache.shape[2])


def _bias_view(bias: torch.Tensor, out_channels: int) -> torch.Tensor:
    if bias.numel() != out_channels:
        raise ValueError(f"Conv bias must contain {out_channels} values, got {bias.shape}")
    return bias.reshape(1, out_channels, 1, 1, 1)


def _effective_padding(
    cache: torch.Tensor,
    pre_padding: tuple[int, int, int],
) -> tuple[int, int, int]:
    cache_t = _cache_frames(cache)
    if cache_t > pre_padding[0]:
        raise ValueError(
            f"Cache length {cache_t} exceeds temporal left padding {pre_padding[0]}"
        )
    return (pre_padding[0] - cache_t, pre_padding[1], pre_padding[2])


def _validate_contract(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> None:
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
        raise ValueError(f"Conv3d padding must be non-negative, got {pre_padding=} {post_padding=}")
    if any(value <= 0 for value in (*stride, *dilation)):
        raise ValueError(f"Conv3d stride/dilation must be positive, got {stride=} {dilation=}")
    if groups != 1:
        raise ValueError(f"Wan VAE fused Concat+Conv requires groups=1, got {groups}")
    if x.dim() != 5:
        raise ValueError(f"Fused Concat+Conv expects 5D input, got {x.shape}")
    if weight.dim() != 5 or tuple(weight.shape[2:]) != (3, 3, 3):
        raise ValueError(f"Wan VAE fused Concat+Conv requires [K,C,3,3,3], got {weight.shape}")
    if int(weight.shape[1]) != int(x.shape[1]):
        raise ValueError(f"Weight/input channel mismatch: {weight.shape=} {x.shape=}")
    if weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("Input and weight must have the same dtype and device")
    if not is_empty_sentinel(cache):
        if cache.dim() != 5:
            raise ValueError(f"Cache must be 5D, got {cache.shape}")
        if (
            int(cache.shape[0]) != int(x.shape[0])
            or int(cache.shape[1]) != int(x.shape[1])
            or int(cache.shape[3]) != int(x.shape[3])
            or int(cache.shape[4]) != int(x.shape[4])
        ):
            raise ValueError(f"Cache spatial/batch/channel shape mismatch: {cache.shape=} {x.shape=}")
        if cache.dtype != x.dtype or cache.device != x.device:
            raise ValueError("Cache and input must have the same dtype and device")
    if not is_empty_sentinel(bias):
        _bias_view(bias, int(weight.shape[0]))
        if bias.dtype != x.dtype or bias.device != x.device:
            raise ValueError("Bias and input must have the same dtype and device")


def _conv_output_shape(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[int, ...]:
    temporal_in = int(x.shape[2]) + _cache_frames(cache)
    input_spatial = (temporal_in, int(x.shape[3]), int(x.shape[4]))
    output_dims = []
    for input_dim, kernel_dim, pre, post, step, dil in zip(
        input_spatial,
        weight.shape[2:],
        pre_padding,
        post_padding,
        stride,
        dilation,
    ):
        output_dims.append((input_dim + pre + post - dil * (kernel_dim - 1) - 1) // step + 1)
    return (int(x.shape[0]), int(weight.shape[0]), *output_dims)


def _reference(
    x: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> torch.Tensor:
    image = x if is_empty_sentinel(cache) else torch.cat((cache, x), dim=2)
    pad = (
        pre_padding[2],
        post_padding[2],
        pre_padding[1],
        post_padding[1],
        pre_padding[0],
        post_padding[0],
    )
    image = F.pad(image, pad)
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
    cache: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    groups: int,
) -> GraphEntry:
    del groups
    hipdnn = require_hipdnn()
    graph = hipdnn.pygraph(
        handle=hipdnn_handle(hipdnn, x.device),
        io_data_type=hipdnn_data_type(hipdnn, x.dtype),
        intermediate_data_type=hipdnn.data_type.FLOAT,
        compute_data_type=hipdnn.data_type.FLOAT,
        name="wan_vae_concat_conv_bias",
    )
    x_tensor = graph.tensor_like(x)
    inputs: dict[str, Any] = {"x": x_tensor}
    if is_empty_sentinel(cache):
        image_tensor = x_tensor
    else:
        cache_tensor = graph.tensor_like(cache)
        inputs["cache"] = cache_tensor
        image_tensor = graph.concatenate(
            x=[cache_tensor, x_tensor], axis=2, name="causal_cache_concat"
        )
    weight_tensor = graph.tensor_like(weight)
    inputs["weight"] = weight_tensor
    conv_tensor = conv_fprop(
        graph,
        image_tensor,
        weight_tensor,
        pre_padding,
        post_padding,
        stride,
        dilation,
        "conv3d",
    )
    if is_empty_sentinel(bias):
        output_tensor = conv_tensor
    else:
        bias_tensor = graph.tensor_like(bias)
        inputs["bias"] = bias_tensor
        output_tensor = graph.add(a=conv_tensor, b=bias_tensor, name="bias")
    output_tensor.set_output(True)
    return make_graph_entry(graph, inputs, output_tensor, x)


@custom_op("turbo_physai::wan_vae_concat_conv_bias", mutates_args=())
def _concat_conv_bias_op(
    x: torch.Tensor,
    cache: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    pre_padding = (pre_t, pre_h, pre_w)
    post_padding = (post_t, post_h, post_w)
    stride = (stride_t, stride_h, stride_w)
    dilation = (dilation_t, dilation_h, dilation_w)
    key = graph_cache_key(
        tensor_signature(x),
        tensor_signature(cache),
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
        lambda: _build_graph(
            x,
            cache,
            weight,
            bias,
            pre_padding,
            post_padding,
            stride,
            dilation,
            groups,
        ),
        what="Concat+Conv3D+bias",
        describe=lambda: (
            f"x={tensor_signature(x)}, cache={tensor_signature(cache)}, "
            f"weight={tensor_signature(weight)}, bias={tensor_signature(bias)}, "
            f"pre_padding={pre_padding}, post_padding={post_padding}, "
            f"stride={stride}, dilation={dilation}, groups={groups}"
        ),
    )
    if entry is None:
        output = _reference(
            x, cache, weight, bias, pre_padding, post_padding, stride, dilation, int(groups)
        )
        return output, _new_cache(x)
    bindings: dict[str, torch.Tensor] = {"x": x, "weight": weight}
    if not is_empty_sentinel(cache):
        bindings["cache"] = cache
    if not is_empty_sentinel(bias):
        bindings["bias"] = bias
    return execute_graph(entry, bindings, x), _new_cache(x)


@_concat_conv_bias_op.register_fake
def _concat_conv_bias_fake(
    x: torch.Tensor,
    cache: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    del bias, groups
    pre_padding = (pre_t, pre_h, pre_w)
    post_padding = (post_t, post_h, post_w)
    stride = (stride_t, stride_h, stride_w)
    dilation = (dilation_t, dilation_h, dilation_w)
    shape = _conv_output_shape(x, cache, weight, pre_padding, post_padding, stride, dilation)
    output = torch.empty(
        shape,
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    return output, _new_cache(x)


def fused_causal_cache_pad_conv3d_ndhwc(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    pre_padding: tuple[int, int, int] = (2, 1, 1),
    post_padding: tuple[int, int, int] = (0, 1, 1),
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse causal cache concat/padding, Conv3d, bias, and cache refresh."""

    if x.dim() != 5:
        raise ValueError(f"Fused causal Conv3d expects a 5D tensor, got {x.shape}")
    cache = empty_sentinel(x) if cache_x is None else cache_x
    bias_view = empty_sentinel(x) if bias is None else _bias_view(bias, int(weight.shape[0]))
    _validate_contract(
        x,
        cache,
        weight,
        bias_view,
        pre_padding,
        post_padding,
        stride,
        dilation,
        groups,
    )
    effective_pre = _effective_padding(cache, pre_padding)
    base_eligible = (
        fused_ops_enabled()
        and fused_dtype_supported(x.dtype)
        and weight.dtype == x.dtype
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and weight.is_contiguous(memory_format=torch.channels_last_3d)
        and (
            is_empty_sentinel(cache)
            or cache.is_contiguous(memory_format=torch.channels_last_3d)
        )
        and (is_empty_sentinel(bias_view) or bias_view.dtype == x.dtype)
        and conv_channels_supported(int(x.shape[1]), int(weight.shape[0]))
        and groups == 1
        and not any(tensor.requires_grad for tensor in (x, cache, weight, bias_view))
    )
    if base_eligible and not fused_concat_enabled():
        image = x
        if not is_empty_sentinel(cache):
            image = torch.cat((cache, x), dim=2).contiguous(
                memory_format=torch.channels_last_3d
            )
        output = fused_conv3d_bias_ndhwc(
            image,
            weight,
            bias,
            pre_padding=effective_pre,
            post_padding=post_padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        return output, _new_cache(x)
    eligible = base_eligible and fused_concat_enabled()
    if not eligible:
        output = _reference(
            x,
            cache,
            weight,
            bias_view,
            effective_pre,
            post_padding,
            stride,
            dilation,
            groups,
        )
        return output, _new_cache(x)

    x = to_canonical_ndhwc(x)
    weight = to_canonical_ndhwc(weight)
    if not is_empty_sentinel(cache):
        cache = to_canonical_ndhwc(cache)
    return _concat_conv_bias_op(
        x,
        cache,
        weight,
        bias_view,
        *effective_pre,
        *post_padding,
        *stride,
        *dilation,
        int(groups),
    )


def fused_concat_conv_bias(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    pre_padding: tuple[int, int, int] = (2, 1, 1),
    post_padding: tuple[int, int, int] = (0, 1, 1),
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    groups: int = 1,
) -> torch.Tensor:
    """Backward-compatible main-output-only wrapper."""

    output, _ = fused_causal_cache_pad_conv3d_ndhwc(
        x,
        cache_x,
        weight,
        bias,
        pre_padding=pre_padding,
        post_padding=post_padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
    return output


__all__ = ["fused_causal_cache_pad_conv3d_ndhwc", "fused_concat_conv_bias"]
