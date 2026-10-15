# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""Instance-local ZeRO-2 sharded reduction with bounded temporary storage.

DeepSpeed keeps gradients in backward order, not destination-rank order. Pack
balanced buckets by destination for native ReduceScatter; use directed Reduce
for skewed buckets rather than transmitting world_size copies of zero padding.
Owner slices receive the reduced result. Non-owner slices are scratch
afterwards: ZeRO-2 reads only the owner partition, and its own allreduce path
overwrites the rest as well.

Nothing is allocated per bucket. Skewed buckets reduce each contiguous owner
range in place when the gradient dtype is the communication dtype; all other
packing goes through one persistent staging buffer per device and dtype.
A device allocation inside the backward hook can block the host for seconds
near the memory limit and stall every rank.
"""

import functools
import inspect
import logging
import types

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_PACK_ELEMENTS = 64 * 1024 * 1024  # Total packed elements, across all ranks.
_staging = {}


def _staging_buffer(device, dtype, numel):
    """Persistent staging storage, grown only when a larger limit is requested.

    Reuse is ordered on the caller's stream: each staged collective is waited on
    with Work.wait() before the next copy into the buffer is enqueued.
    """
    key = (device, dtype)
    buffer = _staging.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = _staging[key] = torch.empty(numel, device=device, dtype=dtype)
    return buffer


def _pieces(tensors, start, count):
    """Views of a logical concatenation; does not allocate or flatten tensors."""
    result = []
    stop = start + count
    offset = 0
    for tensor in tensors:
        end = offset + tensor.numel()
        left, right = max(start, offset), min(stop, end)
        if right > left:
            result.append(tensor.reshape(-1).narrow(0, left - offset, right - left))
        offset = end
        if offset >= stop:
            break
    return result


def _copy_from_flat(targets, source):
    offset = 0
    for target in targets:
        target.copy_(source.narrow(0, offset, target.numel()))
        offset += target.numel()


def reduce_by_owner(
    tensors, owners, *, group, dtype, divide=False, max_elements=_PACK_ELEMENTS
):
    """SUM/average into owner views with async collectives and stream-local waits.

    All ranks must supply identical sizes/owners in the same order. Work.wait()
    establishes the output dependency on the caller's CUDA stream; it does not
    insert a wait onto the independent backward-compute stream. Non-owner views
    may be modified. Calls on one device must share a stream, which orders reuse
    of the staging buffer.
    """
    world, rank = dist.get_world_size(group), dist.get_rank(group)
    if len(tensors) != len(owners) or max_elements < world:
        raise ValueError("Invalid owner reduction plan or temporary-storage limit")
    by_rank = [[] for _ in range(world)]
    for tensor, owner in zip(tensors, owners):
        if not 0 <= owner < world or not tensor.is_contiguous():
            raise ValueError(
                "Owner reduction requires contiguous views and valid group-local ranks"
            )
        by_rank[owner].append(tensor)
    lengths = [sum(t.numel() for t in ts) for ts in by_rank]
    if not sum(lengths):
        return
    device = next(t for t in tensors if t.numel()).device
    maximum = max(lengths)
    if maximum * world <= 2 * sum(lengths):
        width = max_elements // world
        staging = _staging_buffer(device, dtype, world * width + width)
        for start in range(0, maximum, width):
            size = min(width, maximum - start)
            packed = staging.narrow(0, 0, world * size)
            output = staging.narrow(0, world * size, size)
            local_views = []
            for owner, views in enumerate(by_rank):
                pieces = _pieces(
                    views, start, min(size, max(0, lengths[owner] - start))
                )
                offset = owner * size
                for piece in pieces:
                    packed.narrow(0, offset, piece.numel()).copy_(piece)
                    offset += piece.numel()
                if offset < (owner + 1) * size:
                    packed.narrow(0, offset, (owner + 1) * size - offset).zero_()
                if owner == rank:
                    local_views = pieces
            if divide:
                packed.div_(world)
            work = dist.reduce_scatter_tensor(
                output, packed, group=group, async_op=True
            )
            work.wait()
            _copy_from_flat(local_views, output)
    else:
        # A bucket belonging to one owner must not become a W-times-larger RS.
        for owner, views in enumerate(by_rank):
            if not lengths[owner]:
                continue
            dst = dist.get_global_rank(group, owner) if group is not None else owner
            if all(view.dtype == dtype for view in views):
                # In place per contiguous range, like DeepSpeed's allreduce path:
                # no staging copy and no chunking. Callers merge adjacent ranges.
                for view in views:
                    if not view.numel():
                        continue
                    if divide:
                        view.div_(world)
                    work = dist.reduce(view, dst=dst, group=group, async_op=True)
                    work.wait()
                continue
            for start in range(0, lengths[owner], max_elements):
                count = min(max_elements, lengths[owner] - start)
                pieces = _pieces(views, start, count)
                packed = _staging_buffer(device, dtype, max_elements).narrow(
                    0, 0, count
                )
                offset = 0
                for piece in pieces:
                    packed.narrow(0, offset, piece.numel()).copy_(piece)
                    offset += piece.numel()
                if divide:
                    packed.div_(world)
                work = dist.reduce(packed, dst=dst, group=group, async_op=True)
                work.wait()
                if owner == rank:
                    _copy_from_flat(pieces, packed)


def _average_tensor(self, tensor, communication_data_type):
    producer = torch.cuda.current_stream(tensor.device)
    stream = self.reduction_stream
    stream.wait_stream(producer)
    bucket = self.ipg_buckets[communication_data_type]
    ranges = []  # (start, length, owner); adjacent ranges of one owner merged.
    cursor = 0
    for group_id, param_index, param_id in bucket.params:
        param = self.bit16_groups[group_id][param_index]
        # DeepSpeed may store integral partition offsets as floats. Normalize
        # before arithmetic so both narrow's start and length remain integers.
        partitions = sorted(
            (int(self.grad_start_offset[group_id][owner][param_id]), owner)
            for owner in self.param_to_partition_ids[group_id][param_id]
        )
        for index, (offset, owner) in enumerate(partitions):
            stop = (
                partitions[index + 1][0]
                if index + 1 < len(partitions)
                else param.numel()
            )
            start, length = cursor + offset, stop - offset
            if (
                ranges
                and ranges[-1][2] == owner
                and ranges[-1][0] + ranges[-1][1] == start
            ):
                ranges[-1] = (ranges[-1][0], ranges[-1][1] + length, owner)
            else:
                ranges.append((start, length, owner))
        cursor += param.numel()
    if cursor != tensor.numel():
        raise RuntimeError("ZeRO bucket mapping does not cover the reduction tensor")
    # One view per contiguous owner range lets a skewed bucket reduce in place.
    views = [tensor.narrow(0, start, length) for start, length, _ in ranges]
    owners = [owner for _, _, owner in ranges]
    with torch.cuda.stream(stream):
        tensor.record_stream(stream)
        # Preserve DeepSpeed's existing pre-reduction rounding/scaling order.
        tensor.div_(dist.get_world_size(self.dp_process_group))
        reduce_by_owner(
            views, owners, group=self.dp_process_group, dtype=communication_data_type
        )


def validate_zero(optimizer):
    """Reject unsupported modes before replacing any instance method."""
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    if type(optimizer) is not DeepSpeedZeroOptimizer:
        raise ValueError(
            "Online optimizations require native DeepSpeedZeroOptimizer (ZeRO-2)"
        )
    if (
        not optimizer.partition_gradients
        or optimizer.cpu_offload
        or optimizer.has_moe_layers
    ):
        raise ValueError(
            "Online optimizations require ZeRO-2 without CPU offload or MoE"
        )
    if getattr(optimizer, "zenflow", False) or optimizer.sequence_parallel_size != 1:
        raise ValueError(
            "Online optimizations do not support ZenFlow or sequence parallelism"
        )
    if any(
        group is not optimizer.dp_process_group
        for group in optimizer.real_dp_process_group
    ):
        raise ValueError("Online optimizations require one data-parallel process group")


def step_with_gather(optimizer, gather):
    """A bound copy of optimizer.step whose step-end all_gather_dp_groups is ``gather``.

    Only this instance's step sees the replacement; DeepSpeed's module globals
    and other callers (update_lp_params, checkpoint loading) are unchanged.
    """
    function = optimizer.step.__func__
    if "all_gather_dp_groups" not in function.__code__.co_names:
        raise RuntimeError(
            "Unsupported DeepSpeed step: no direct all_gather_dp_groups call"
        )
    expected = (
        "groups_flat",
        "partitioned_param_groups",
        "dp_process_group",
        "start_alignment_factor",
        "allgather_bucket_size",
    )
    if (
        tuple(
            inspect.signature(function.__globals__["all_gather_dp_groups"]).parameters
        )
        != expected
    ):
        raise RuntimeError("Unsupported DeepSpeed AllGather interface")
    namespace = dict(function.__globals__)
    namespace["all_gather_dp_groups"] = gather
    replacement = types.FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    replacement.__kwdefaults__ = function.__kwdefaults__
    functools.update_wrapper(replacement, function)
    return types.MethodType(replacement, optimizer)


def install_reduce_scatter(optimizer):
    validate_zero(optimizer)
    if getattr(optimizer, "_openwam_reduce_scatter", False):
        return
    if not (
        optimizer.overlap_comm
        and optimizer.contiguous_gradients
        and optimizer.reduce_scatter
    ):
        raise ValueError(
            "Owner reduction requires contiguous gradients, reduce_scatter and overlap_comm"
        )
    if not optimizer.postscale_gradients or optimizer.gradient_predivide_factor != 1.0:
        raise ValueError(
            "Owner reduction requires standard postscale/predivide settings"
        )
    for name, parameters in (
        ("average_tensor", ("tensor", "communication_data_type")),
        ("reduce_ipg_grads", ("comm_dtype",)),
        ("reduce_independent_p_g_buckets_and_remove_grads", ("param", "i")),
    ):
        if tuple(inspect.signature(getattr(optimizer, name)).parameters) != parameters:
            raise RuntimeError(f"Unsupported DeepSpeed interface: {name}")
    reduced = optimizer.reduce_ipg_grads
    accumulate = optimizer.reduce_independent_p_g_buckets_and_remove_grads
    events = {}

    def reduce_ipg(this, comm_dtype=None):
        # Capture indices before DeepSpeed clears each bucket's metadata.
        keys = [
            (dtype, bucket.index)
            for dtype, bucket in this.ipg_buckets.items()
            if (comm_dtype is None or dtype == comm_dtype) and bucket.elements
        ]
        result = reduced(comm_dtype=comm_dtype)
        for key in keys:
            event = torch.cuda.Event()
            event.record(this.reduction_stream)  # Includes copy_grads_in_partition.
            events[key] = event
        return result

    def accumulate_grad(this, param, i):
        dtype = this.get_param_comm_dtype(param)
        bucket = this.ipg_buckets[dtype]
        if bucket.elements + param.numel() > this.reduce_bucket_size:
            # Wait only when the *other* double-buffer is about to be reused.
            # Compute does not wait here for the bucket being reduced right now.
            event = events.get((dtype, 1 - bucket.index))
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
        return accumulate(param, i)

    optimizer.average_tensor = types.MethodType(_average_tensor, optimizer)
    optimizer.reduce_ipg_grads = types.MethodType(reduce_ipg, optimizer)
    optimizer.reduce_independent_p_g_buckets_and_remove_grads = types.MethodType(
        accumulate_grad, optimizer
    )
    optimizer._openwam_reduce_scatter = True
    logger.info(
        "[optimizations] async owner reduction: native ReduceScatter / sparse-owner Reduce"
    )
