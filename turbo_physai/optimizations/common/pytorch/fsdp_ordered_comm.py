# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ordered FSDP collective submission streams for HIP/RCCL.

FSDP2 can submit collectives from different CUDA streams.  RCCL requires
collectives on one communicator to be submitted in the same global order, so
the HCU path hands each process group's collective work to one canonical
stream and synchronizes the caller stream around it.

This module is model-agnostic.  When an adapted Cosmos source tree already
provides ``cosmos_framework.utils.fsdp_ordered_comm``, model adapters should
import from there instead so every collective shares one stream table; this
module is the implementation used on a clean upstream tree.
"""

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch
import torch.distributed as dist

try:  # torch >= 2.4 public path (target: torch 2.10 HIP)
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import Partial, Replicate
except ImportError:  # older local dev environments (torch <= 2.3)
    from torch.distributed._tensor import DTensor
    from torch.distributed._tensor.placement_types import Replicate

    try:
        from torch.distributed._tensor.placement_types import Partial
    except ImportError:
        from torch.distributed._tensor.placement_types import _Partial as Partial


_streams: dict[tuple[str, int, dist.ProcessGroup], torch.cuda.Stream] = {}
_streams_lock = threading.RLock()


def _group_key(group: dist.ProcessGroup, device: torch.device) -> tuple[str, int, dist.ProcessGroup]:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    return device.type, device_index, group


def _get_group_stream(group: dist.ProcessGroup, device: torch.device) -> torch.cuda.Stream:
    key = _group_key(group, device)
    with _streams_lock:
        stream = _streams.get(key)
        if stream is None:
            with torch.cuda.device(key[1]):
                stream = torch.cuda.Stream(device=key[1])
            _streams[key] = stream
        return stream


@contextmanager
def _ordered_group_stream(group: dist.ProcessGroup, device: torch.device) -> Iterator[None]:
    """Run one collective on the process group's canonical submission stream."""

    caller_stream = torch.cuda.current_stream(device)
    collective_stream = _get_group_stream(group, device)
    with _streams_lock:
        collective_stream.wait_stream(caller_stream)
        with torch.cuda.stream(collective_stream):
            yield
        caller_stream.wait_stream(collective_stream)


class OrderedAllGather:
    """FSDP2 all-gather implementation with per-group stream ordering."""

    def allocate(
        self,
        size: Sequence["int | torch.SymInt"],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
    ) -> "dist.Work | None":
        with _ordered_group_stream(group, input_tensor.device):
            return dist.all_gather_into_tensor(
                output_tensor,
                input_tensor,
                group=group,
                async_op=async_op,
            )


class OrderedReduceScatter:
    """FSDP2 reduce-scatter implementation with per-group stream ordering."""

    def allocate(
        self,
        size: Sequence["int | torch.SymInt"],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.empty(*size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: dist.ReduceOp,
        async_op: bool = False,
    ) -> "dist.Work | None":
        with _ordered_group_stream(group, input_tensor.device):
            return dist.reduce_scatter_tensor(
                output_tensor,
                input_tensor,
                group=group,
                op=op,
                async_op=async_op,
            )


def ordered_all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: "dist.ProcessGroup | None" = None,
    async_op: bool = False,
) -> "dist.Work | None":
    """Submit an all-reduce on the process group's canonical stream."""

    if group is None:
        group = dist.group.WORLD
    with _ordered_group_stream(group, tensor.device):
        return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def materialize_dtensor_norm(norm: DTensor) -> torch.Tensor:
    """Reduce a norm DTensor while preserving per-communicator stream order."""

    local_norm = norm.to_local()
    mesh = norm.device_mesh
    for mesh_dim, placement in enumerate(norm.placements):
        if isinstance(placement, Replicate):
            continue
        if not isinstance(placement, Partial):
            raise RuntimeError(f"Unexpected gradient norm DTensor placement: {placement}")
        group = mesh.get_group(mesh_dim)
        with _ordered_group_stream(group, local_norm.device):
            local_norm = placement._reduce_value(local_norm, mesh, mesh_dim)
    return local_norm
