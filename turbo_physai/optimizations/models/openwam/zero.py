# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
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

import contextlib
import functools
import inspect
import logging
import types

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)
_PACK_ELEMENTS = 64 * 1024 * 1024  # Total packed elements, across all ranks.
_staging = {}


class _BucketCopies:
    """Bound source lifetimes while coalescing producer-stream bucket copies."""

    def __init__(self, max_bytes=64 * 1024 * 1024, max_tensors=32):
        self.max_bytes = max_bytes
        self.max_tensors = max_tensors
        self.dsts, self.srcs = [], []
        self.bytes = 0
        self.stream = None

    def append(self, dst, src):
        stream = torch.cuda.current_stream(src.device) if src.is_cuda else None
        # A small view can pin a much larger original gradient allocation.
        # Conservatively count its storage, even if two views share it.
        size = src.untyped_storage().nbytes()
        if self.srcs and (stream != self.stream or self.bytes + size > self.max_bytes):
            self.flush()
        # A large gradient cannot share the bounded batch. Avoid retaining it
        # and avoid replacing one DMA copy with many foreach kernels.
        if size >= self.max_bytes:
            dst.copy_(src)
            return
        self.stream = stream
        self.dsts.append(dst)
        # This view retains the original storage across grad_reduc.data rebinding.
        self.srcs.append(src.detach())
        self.bytes += size
        if len(self.srcs) >= self.max_tensors or self.bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        if not self.srcs:
            return
        stream = self.stream
        try:
            with torch.cuda.stream(
                stream
            ) if stream is not None else contextlib.nullcontext():
                if len(self.srcs) == 1:
                    self.dsts[0].copy_(self.srcs[0])
                else:
                    torch._foreach_copy_(self.dsts, self.srcs)
            if stream is not None:
                current = torch.cuda.current_stream(stream.device)
                if current != stream:
                    # A later average_tensor waits on its current producer;
                    # include a batch queued on an earlier producer stream.
                    current.wait_stream(stream)
        finally:
            self.dsts.clear()
            self.srcs.clear()
            self.bytes = 0
            self.stream = None


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
            if len(views) == 1 and views[0].dtype == dtype:
                view = views[0]
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
                if all(piece.dtype == dtype for piece in pieces):
                    dst_views = []
                    offset = 0
                    for piece in pieces:
                        dst_views.append(packed.narrow(0, offset, piece.numel()))
                        offset += piece.numel()
                    torch._foreach_copy_(dst_views, pieces)
                else:
                    offset = 0
                    for piece in pieces:
                        packed.narrow(0, offset, piece.numel()).copy_(piece)
                        offset += piece.numel()
                if divide:
                    packed.div_(world)
                work = dist.reduce(packed, dst=dst, group=group, async_op=True)
                work.wait()
                if owner == rank:
                    if all(piece.dtype == dtype for piece in pieces):
                        torch._foreach_copy_(pieces, dst_views)
                    else:
                        _copy_from_flat(pieces, packed)


def _average_tensor(self, tensor, communication_data_type):
    producer = torch.cuda.current_stream(tensor.device)
    stream = self.reduction_stream
    stream.wait_stream(producer)
    bucket = self.ipg_buckets[communication_data_type]
    cache = getattr(self, "_openwam_bucket_plan_cache", None)
    if cache is None:
        cache = self._openwam_bucket_plan_cache = {}
    cache_key = (
        tuple(bucket.params),
        int(tensor.numel()),
        int(self.nccl_start_alignment_factor),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        ranges, owners = cached
    else:
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
            raise RuntimeError(
                "ZeRO bucket mapping does not cover the reduction tensor"
            )
        owners = [owner for _, _, owner in ranges]
        cache[cache_key] = (ranges, owners)
    # One view per contiguous owner range lets a skewed bucket reduce in place.
    views = [tensor.narrow(0, start, length) for start, length, _ in ranges]
    with torch.cuda.stream(stream):
        tensor.record_stream(stream)
        fused = getattr(self, "_openwam_scaled_pack", None)
        if fused is not None and fused.run(
            tensor,
            ranges,
            group=self.dp_process_group,
            dtype=communication_data_type,
            max_elements=_PACK_ELEMENTS,
            staging_buffer=_staging_buffer,
        ):
            return
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
    events = {}
    bucket_copies = {}

    orig_copy_grads = optimizer.copy_grads_in_partition
    orig_get_grad_norm_direct = optimizer.get_grad_norm_direct
    orig_update_model_weights = optimizer._update_model_bit16_weights
    from turbo_physai.optimizations.models.openwam.settings import enabled

    if enabled("zero_scaled_pack"):
        from turbo_physai.optimizations.models.openwam.zero_fused import ScaledPack

        optimizer._openwam_scaled_pack = ScaledPack()
    norm_overlap = None
    if enabled("zero_norm_overlap"):
        from turbo_physai.optimizations.models.openwam.zero_norm import NormOverlap

        norm_overlap = NormOverlap(optimizer)
        norm_overlap.install()
        optimizer._openwam_norm_overlap = norm_overlap
    pending_dsts = []
    pending_srcs = []
    in_reduce_ipg = False

    def reduce_ipg(this, comm_dtype=None):
        nonlocal in_reduce_ipg
        # Flush before DeepSpeed/average_tensor records the producer dependency.
        for dtype, copies in bucket_copies.items():
            if comm_dtype is None or dtype == comm_dtype:
                copies.flush()
        # Capture indices before DeepSpeed clears each bucket's metadata.
        keys = [
            (dtype, bucket.index)
            for dtype, bucket in this.ipg_buckets.items()
            if (comm_dtype is None or dtype == comm_dtype) and bucket.elements
        ]
        in_reduce_ipg = True
        try:
            result = reduced(comm_dtype=comm_dtype)
            if pending_dsts:
                with torch.cuda.stream(this.reduction_stream):
                    torch._foreach_copy_(pending_dsts, pending_srcs)
        finally:
            in_reduce_ipg = False
            pending_dsts.clear()
            pending_srcs.clear()
        for key in keys:
            event = torch.cuda.Event()
            event.record(this.reduction_stream)  # Includes copy_grads_in_partition.
            events[key] = event
        return result

    def accumulate_grad(this, param, i):
        dtype = this.get_param_comm_dtype(param)
        bucket = this.ipg_buckets[dtype]
        numel = param.numel()
        if bucket.elements + numel > this.reduce_bucket_size:
            this.reduce_ipg_grads(comm_dtype=dtype)
            # Wait only when the *other* double-buffer is about to be reused.
            # Compute does not wait here for the bucket being reduced right now.
            event = events.get((dtype, 1 - bucket.index))
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            bucket.index = 1 - bucket.index
        if not getattr(param, "ds_grad_is_ready", True):
            return
        param_id = this.get_param_id(param)
        if this.params_already_reduced[param_id]:
            raise RuntimeError(
                f"Gradient for parameter {param_id} was reduced twice in one step"
            )
        grad = this.get_gradient_for_reduction(param)
        if numel > this.reduce_bucket_size:
            this.extra_large_param_to_reduce[dtype] = param
        else:
            dst = bucket.buffer[bucket.index].narrow(0, bucket.elements, numel)
            copies = bucket_copies.get(dtype)
            if copies is None:
                copies = bucket_copies[dtype] = _BucketCopies()
            copies.append(dst, grad.reshape(-1))
            grad.data = dst.view_as(grad)
        bucket.elements += numel
        bucket.grads.append(grad)
        bucket.params.append((i, param.param_idx_in_group, param_id))

    def copy_grads(this, param):
        if not in_reduce_ipg or this.cpu_offload or this.grads_in_partition is None:
            return orig_copy_grads(param)
        grad_reduc = this.get_gradient_for_reduction(param)
        numel = param.numel()
        offset = this.grads_in_partition_offset
        this.grads_in_partition_offset = offset + numel
        dst_view = (
            this.grads_in_partition.view(-1)
            .narrow(0, offset, numel)
            .view_as(grad_reduc)
        )
        pending_dsts.append(dst_view)
        pending_srcs.append(grad_reduc.view_as(grad_reduc))
        grad_reduc.data = dst_view

    def get_grad_norm(this, gradients, params, norm_type=2):
        norm_type = float(norm_type)
        if norm_type != 2.0 or getattr(this, "cpu_offload", False):
            return orig_get_grad_norm_direct(gradients, params, norm_type)
        total_norm = (
            norm_overlap.take(gradients, params) if norm_overlap is not None else None
        )
        if total_norm is None:
            from turbo_physai.optimizations.models.openwam.zero_norm import (
                selected_gradients,
            )

            grads_for_norm = selected_gradients(this, gradients, params)
            if not grads_for_norm:
                total_norm = torch.zeros((), dtype=torch.float32, device=this.device)
            else:
                try:
                    norms = torch._foreach_norm(
                        grads_for_norm, 2.0, dtype=torch.float64
                    )
                    total_norm = torch.stack(norms).square().sum().float()
                except Exception:
                    return orig_get_grad_norm_direct(gradients, params, norm_type)
        dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=this.dp_process_group)
        if hasattr(this, "_model_parallel_all_reduce"):
            this._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)
        total_norm = total_norm.pow(1.0 / norm_type)
        total_norm.masked_fill_(torch.isinf(total_norm) | torch.isnan(total_norm), -1.0)
        return total_norm

    def update_model_weights(this, group_index):
        flat = this.bit16_groups_flat[group_index]
        params = this.bit16_groups[group_index]
        offset = flat.storage_offset()
        pointer = flat.untyped_storage().data_ptr()
        # Same allocation alone is insufficient: swapped or shifted parameter
        # views must be rebound by DeepSpeed, too. Check every view, not only
        # the first one, so external parameter replacement cannot stale a cache.
        for param in params:
            data = param.data
            if (
                not data.is_contiguous()
                or data.dtype != flat.dtype
                or data.device != flat.device
                or data.storage_offset() != offset
                or data.untyped_storage().data_ptr() != pointer
            ):
                return orig_update_model_weights(group_index)
            offset += data.numel()
        if not params:
            return orig_update_model_weights(group_index)

    optimizer.copy_grads_in_partition = types.MethodType(copy_grads, optimizer)
    optimizer.get_grad_norm_direct = types.MethodType(get_grad_norm, optimizer)
    optimizer._update_model_bit16_weights = types.MethodType(
        update_model_weights, optimizer
    )
    optimizer.average_tensor = types.MethodType(_average_tensor, optimizer)
    optimizer.reduce_ipg_grads = types.MethodType(reduce_ipg, optimizer)
    optimizer.reduce_independent_p_g_buckets_and_remove_grads = types.MethodType(
        accumulate_grad, optimizer
    )
    optimizer._openwam_reduce_scatter = True
