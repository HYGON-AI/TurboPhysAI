# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small, lazy hipDNN graph helpers for the pure Wan VAE fused operators.

The HCU runtime is not available in the development workspace. Consequently
this module must remain importable on CPU and NVIDIA hosts. It only imports
the HCU ``hipdnn`` Python binding from an enabled operator implementation.
"""

from __future__ import annotations

import importlib
import inspect
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

import logging

log = logging.getLogger(__name__)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_GRAPH_LOCK = threading.RLock()
_HANDLE_CACHE: dict[tuple[str, int, int], Any] = {}
_WARNED_BUILD_FAILURES: set[tuple[Any, ...]] = set()


@dataclass
class GraphEntry:
    """A compiled graph and the tensor descriptors it expects at execution."""

    graph: Any
    input_tensors: Mapping[str, Any]
    output_tensor: Any
    output_shape: tuple[int, ...]
    workspace: torch.Tensor


def fused_ops_enabled() -> bool:
    """Return whether the HCU pure fused path was explicitly requested.

    Controlled by environment variable COSMOS_WAN_VAE_CLEAN_FUSED_OP=1
    (or legacy COSMOS_WAN_VAE_FUSED_OPS=1). Default is 0 (disabled).
    """

    val = os.environ.get("COSMOS_WAN_VAE_CLEAN_FUSED_OP", "").strip().lower()
    if val:
        return val in _TRUE_VALUES
    val_legacy = os.environ.get("COSMOS_WAN_VAE_FUSED_OPS", "0").strip().lower()
    return val_legacy in _TRUE_VALUES


def fused_concat_enabled() -> bool:
    """Whether the Concat+Conv fusion (requirement two) may be used.

    Setting ``COSMOS_WAN_VAE_FUSED_CONCAT=0`` falls back to an explicit
    ``torch.cat`` feeding the Conv3D+bias graph.
    """

    value = os.environ.get("COSMOS_WAN_VAE_FUSED_CONCAT", "1").strip().lower()
    return value in _TRUE_VALUES


def get_custom_op_decorator():
    """Return torch.library.custom_op or a safe stub for torch < 2.4 / local test."""

    if hasattr(torch.library, "custom_op"):
        return torch.library.custom_op

    def custom_op_fallback(name: str, *, mutates_args=()):
        def decorator(fn):
            fn.register_fake = lambda fake_fn: fake_fn
            return fn

        return decorator

    return custom_op_fallback


def require_hipdnn() -> Any:
    """Import hipDNN only for an explicitly enabled HCU execution."""

    try:
        return importlib.import_module("hipdnn")
    except Exception as exc:  # pragma: no cover - exercised only on HCU hosts
        raise RuntimeError(
            "COSMOS_WAN_VAE_CLEAN_FUSED_OP=1 requires an importable HCU hipdnn binding"
        ) from exc


def resolved_device_index(device: torch.device) -> int:
    """Return a concrete device index so ``cuda`` and ``cuda:0`` share a key."""

    if device.index is not None:
        return int(device.index)
    if device.type == "cuda":
        return int(torch.cuda.current_device())
    return -1


def hipdnn_handle(hipdnn: Any, device: torch.device) -> Any:
    """Return one hipDNN handle per (device, thread)."""

    device_index = resolved_device_index(device)
    key = (device.type, device_index, threading.get_ident())
    with _GRAPH_LOCK:
        handle = _HANDLE_CACHE.get(key)
        if handle is not None:
            return handle

        if device.type == "cuda":
            with torch.cuda.device(device_index):
                handle = hipdnn.create_handle()
        else:
            handle = hipdnn.create_handle()
        _HANDLE_CACHE[key] = handle
        return handle


def hipdnn_data_type(hipdnn: Any, dtype: torch.dtype) -> Any:
    """Map the PyTorch dtype used by the VAE to a hipDNN enum value."""

    names_by_dtype = {
        torch.float16: ("HALF", "FLOAT16"),
        torch.bfloat16: ("BFLOAT16", "BFLOAT"),
        torch.float32: ("FLOAT", "FLOAT32"),
    }
    for name in names_by_dtype.get(dtype, ()):
        value = getattr(hipdnn.data_type, name, None)
        if value is not None:
            return value
    raise TypeError(f"Unsupported dtype for hipDNN fused VAE op: {dtype}")


def tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return a hashable descriptor signature for graph-plan caching."""

    return (
        tuple(int(size) for size in tensor.shape),
        tuple(int(stride) for stride in tensor.stride()),
        tensor.dtype,
        tensor.device.type,
        resolved_device_index(tensor.device),
    )


def graph_cache_key(*parts: Any) -> tuple[Any, ...]:
    """Key a compiled-graph cache entry by the thread that will execute it."""

    return (threading.get_ident(), *parts)


def fused_dtype_supported(dtype: torch.dtype) -> bool:
    """Whether the HCU fused Conv3D graphs have engines for *dtype* (BF16 only)."""

    return dtype == torch.bfloat16


def conv_channels_supported(in_channels: int, out_channels: int) -> bool:
    """Channel whitelist for the pure VAE fused operator.

    Only core channels {160, 320, 640} with 16-byte alignment are supported.
    Input Stem (C_in=12) is explicitly rejected to prevent hipDNN from
    selecting the degenerate wg64x64_tt4x4 tile (+885 ms penalty).
    """

    if in_channels < 16 or in_channels % 16 != 0:
        return False
    if in_channels not in {160, 320, 640}:
        return False
    return True


def canonical_ndhwc_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the NDHWC stride the Wan-VAE operator contract mandates.

    The requirement doc fixes this as ``[C*T*H*W, 1, H*W*C, W*C, C]``.
    """

    _, c, d, h, w = shape
    return (c * d * h * w, 1, h * w * c, w * c, c)


def is_canonical_ndhwc(tensor: torch.Tensor) -> bool:
    """Whether *tensor* carries exactly the contract stride."""

    return tensor.dim() == 5 and tuple(tensor.stride()) == canonical_ndhwc_stride(
        tuple(tensor.shape)
    )


def to_canonical_ndhwc(tensor: torch.Tensor) -> torch.Tensor:
    """Materialise *tensor* with the contract stride, copying only when needed."""

    if is_canonical_ndhwc(tensor):
        return tensor
    out = torch.empty(
        tensor.shape,
        dtype=tensor.dtype,
        device=tensor.device,
        memory_format=torch.channels_last_3d,
    )
    out.copy_(tensor)
    return out


def restride_canonical_ndhwc(tensor: torch.Tensor) -> torch.Tensor:
    """Re-describe *tensor* with the contract stride without copying."""

    if is_canonical_ndhwc(tensor):
        return tensor
    shape = tuple(tensor.shape)
    if len(shape) == 5 and shape[0] == 1 and shape[2] == 1 and shape[3] == 1 and shape[4] == 1:
        return tensor.as_strided(shape, canonical_ndhwc_stride(shape))
    return to_canonical_ndhwc(tensor)


def is_empty_sentinel(tensor: torch.Tensor) -> bool:
    """Whether *tensor* is the zero-element optional-input sentinel."""

    return tensor.numel() == 0


def empty_sentinel(like: torch.Tensor) -> torch.Tensor:
    """Create a device-local zero-element tensor for custom-op optional inputs."""

    return like.new_empty((0,))


def get_or_build_graph(
    cache: dict[tuple[Any, ...], GraphEntry | None],
    key: tuple[Any, ...],
    builder: Callable[[], GraphEntry],
    *,
    what: str,
    describe: Callable[[], str],
) -> GraphEntry | None:
    """Build one immutable plan per shape/layout/device/thread signature."""

    if key in cache:
        return cache[key]
    with _GRAPH_LOCK:
        if key in cache:
            return cache[key]
        try:
            entry: GraphEntry | None = builder()
        except RuntimeError as exc:
            entry = None
            if key not in _WARNED_BUILD_FAILURES:
                _WARNED_BUILD_FAILURES.add(key)
                log.warning(
                    f"hipDNN {what} graph build failed; falling back to the unfused "
                    f"reference for this descriptor set. {describe()}: {exc}"
                )
        cache[key] = entry
        return entry


def make_graph_entry(
    graph: Any,
    input_tensors: Mapping[str, Any],
    output_tensor: Any,
    like: torch.Tensor,
) -> GraphEntry:
    """Finish a graph and allocate its persistent workspace."""

    graph.build(hipdnn_handle(require_hipdnn(), like.device))
    workspace_size = int(graph.get_workspace_size())
    workspace = torch.empty((workspace_size,), dtype=torch.uint8, device=like.device)
    output_shape = tuple(int(size) for size in output_tensor.get_dim())
    return GraphEntry(graph, input_tensors, output_tensor, output_shape, workspace)


def execute_graph(
    entry: GraphEntry,
    bindings: Mapping[str, torch.Tensor],
    like: torch.Tensor,
) -> torch.Tensor:
    """Allocate the output and execute a cached hipDNN graph."""

    memory_format = torch.channels_last_3d if len(entry.output_shape) == 5 else torch.contiguous_format
    output = torch.empty(
        entry.output_shape,
        dtype=like.dtype,
        device=like.device,
        memory_format=memory_format,
    )
    variant_pack = {
        entry.input_tensors[name]: tensor.data_ptr()
        for name, tensor in bindings.items()
    }
    variant_pack[entry.output_tensor] = output.data_ptr()
    entry.graph.exec(variant_pack=variant_pack, workspace=entry.workspace.data_ptr())
    return output


_ASYMMETRIC_PADDING_SUPPORT: bool | None = None


def _supports_asymmetric_padding(graph: Any) -> bool:
    """Detect ``pre_padding``/``post_padding`` support without mutating a graph."""

    global _ASYMMETRIC_PADDING_SUPPORT
    if _ASYMMETRIC_PADDING_SUPPORT is not None:
        return _ASYMMETRIC_PADDING_SUPPORT

    method = graph.conv_fprop
    supported = False
    try:
        supported = "pre_padding" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        supported = "pre_padding" in (getattr(method, "__doc__", None) or "")
    _ASYMMETRIC_PADDING_SUPPORT = supported
    return supported


def conv_fprop(
    graph: Any,
    image: Any,
    weight: Any,
    pre_padding: tuple[int, int, int],
    post_padding: tuple[int, int, int],
    stride: tuple[int, int, int],
    dilation: tuple[int, int, int],
    name: str,
) -> Any:
    """Add a convolution, using asymmetric pre/post padding when available."""

    if _supports_asymmetric_padding(graph):
        return graph.conv_fprop(
            image=image,
            weight=weight,
            pre_padding=list(pre_padding),
            post_padding=list(post_padding),
            stride=list(stride),
            dilation=list(dilation),
            name=name,
        )
    if pre_padding != post_padding:
        raise RuntimeError(
            "The installed hipDNN Python binding does not expose asymmetric "
            "pre_padding/post_padding required by causal Conv3d"
        )
    return graph.conv_fprop(
        image=image,
        weight=weight,
        padding=list(pre_padding),
        stride=list(stride),
        dilation=list(dilation),
        name=name,
    )
