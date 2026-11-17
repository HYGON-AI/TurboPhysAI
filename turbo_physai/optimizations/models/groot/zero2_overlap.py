# Adapted from Isaac-GR00T, SPDX-License-Identifier: Apache-2.0
"""Experimental ZeRO-2 overlap for the pinned DTK DeepSpeed build."""
from __future__ import annotations
import inspect
import hashlib
import logging
from functools import wraps
from importlib.metadata import version
from typing import Any
import torch
import torch.distributed as dist
from turbo_physai import CompatibilityResult
from turbo_physai.engine.checking.evidence import ast_hash

logger = logging.getLogger(__name__)
_SUPPORTED_TORCH = "2.10.0+das.opt1.dtk2604.2608131652.g28498c"
_SUPPORTED_DS = "0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df"
_HELPERS = {
    "reduce_ipg_grads": "ast-v1:c69b0884adc264c6cf4fa9af603a8962ac4cca8badeb0a96bf34e8dbda74d2f3",
    "allreduce_and_scatter": "ast-v1:2551da0e64660b3f0d88e079190b9dd0174d9b8eba35d57868def78ed6172745",
    "get_gradient_for_reduction": "ast-v1:98aa0b74f53491366e9da11400fadbfd2525ca1489be36e63f572454f64c6295",
    "get_param_comm_dtype": "ast-v1:0de1d780a5e6b4bd72a502a4c4e765d12a63e1980f4f44408a634454cf96331f",
}

class BucketEventScheduler:
    """Track when each half of a double-buffered gradient bucket is free for reuse.

    The reduction stream records an event once a buffer has been fully consumed. The compute
    stream waits on that event only when it is about to write the same buffer again, instead of
    waiting for every reduction as DeepSpeed's second ``wait_stream`` does.
    """

    def __init__(self, current_stream: Callable[[], Any], event_factory: Callable[[], Any]):
        self._current_stream = current_stream
        self._event_factory = event_factory
        self._events: dict[Hashable, Any] = {}
        self.records = 0
        self.waits = 0

    def reset(self) -> None:
        self._events.clear()

    def record_consumed(self, key: Hashable, reduction_stream: Any) -> None:
        event = self._event_factory()
        event.record(reduction_stream)
        self._events[key] = event
        self.records += 1

    def wait_before_reuse(self, key: Hashable) -> bool:
        event = self._events.pop(key, None)
        if event is None:
            return False
        self._current_stream().wait_event(event)
        self.waits += 1
        return True


def build_rank_and_offsets(optimizer: Any, bucket: Any) -> tuple[list[tuple[int, int, int]], int]:
    """Map a bucket to (owner, offset, numel) slices, merging adjacent runs with the same owner."""
    result: list[tuple[int, int, int]] = []
    current = 0
    previous_owner = None
    for group_index, param_index, param_id in bucket.params:
        param = optimizer.bit16_groups[group_index][param_index]
        partitions = sorted(
            ((pid, optimizer.grad_start_offset[group_index][pid][param_id])
             for pid in optimizer.param_to_partition_ids[group_index][param_id]),
            key=lambda item: item[1],
        )
        for position, (owner, offset) in enumerate(partitions):
            if position == len(partitions) - 1:
                numel = param.numel() - offset
            else:
                numel = partitions[position + 1][1] - offset
            if owner == previous_owner:
                last_owner, last_offset, last_numel = result[-1]
                result[-1] = (last_owner, last_offset, last_numel + numel)
            else:
                result.append((owner, current, numel))
            current += numel
            previous_owner = owner
    return result, current


class _OverlapImplementation:
    def reduce_independent_p_g_buckets_and_remove_grads(self, param: Any, i: int) -> None:
        optimizer = self.optimizer
        grad_reduc = optimizer.get_gradient_for_reduction(param)
        comm_dtype = optimizer.get_param_comm_dtype(param)
        bucket = optimizer.ipg_buckets[comm_dtype]
        if bucket.elements + param.numel() > optimizer.reduce_bucket_size:
            optimizer.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())
            reduced_index = bucket.index
            optimizer.reduce_ipg_grads()
            if optimizer.contiguous_gradients and optimizer.overlap_comm:
                # bucket.clear() reset the index; alternate halves explicitly so compute never
                # writes the half that is still being reduced.
                self.scheduler.record_consumed((comm_dtype, reduced_index), optimizer.reduction_stream)
                bucket.index = 1 - reduced_index
                self.scheduler.wait_before_reuse((comm_dtype, bucket.index))
            optimizer.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

        param_id = optimizer.get_param_id(param)
        assert optimizer.params_already_reduced[param_id] == False, (
            f"Gradient computed twice for this partition (param {param_id}); "
            "multiple gradient reductions are currently not supported"
        )

        if optimizer.contiguous_gradients:
            if param.numel() > optimizer.reduce_bucket_size:
                optimizer.extra_large_param_to_reduce[comm_dtype] = param
            else:
                new_grad_tensor = bucket.buffer[bucket.index].narrow(0, bucket.elements, param.numel())
                new_grad_tensor.copy_(grad_reduc.view(-1))
                grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)

        bucket.elements += param.numel()

        assert grad_reduc is not None, f"Invalid to reduce param {param_id} with None gradient"

        if getattr(param, "ds_grad_is_ready", True):
            bucket.grads.append(grad_reduc)
            bucket.params.append((i, param.param_idx_in_group, param_id))

        from deepspeed.moe.utils import is_moe_param

        if is_moe_param(param):
            bucket.has_moe_params = True

        optimizer.report_ipg_memory_usage("End ipg_remove_grads", 0)

    def average_tensor(self, tensor, communication_data_type):
        optimizer = self.optimizer
        current = self.accelerator.current_stream()
        stream = optimizer.reduction_stream
        stream.wait_stream(current)
        tensor.record_stream(stream)
        with self.accelerator.stream(stream):
            bucket = optimizer.ipg_buckets[communication_data_type]
            if bucket.has_moe_params:
                raise RuntimeError("MoE parameters are unsupported")
            spec, total = build_rank_and_offsets(optimizer, bucket)
            if total != tensor.numel():
                raise RuntimeError(f"Bucket layout covers {total} elements but tensor has {tensor.numel()}")
            group = optimizer.dp_process_group
            tensor.div_(dist.get_world_size(group=group) / float(optimizer.sequence_parallel_size))
            slices = [tensor.narrow(0, int(offset), int(numel)) for _, offset, numel in spec]
            state = getattr(optimizer, "_turbo_physai_gro", None)
            previous = state.get("in_average", False) if state is not None else False
            if state is not None:
                state["in_average"] = True
            try:
                optimizer.allreduce_and_scatter(list(zip((owner for owner, _, _ in spec), slices)),
                    communication_data_type, numel_per_bucket=optimizer.reduce_bucket_size,
                    divide=False, process_group=group)
            finally:
                if state is not None:
                    state["in_average"] = previous


def vendor_helpers_check(context):
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        actual = {name: ast_hash(getattr(DeepSpeedZeroOptimizer, name)) for name in _HELPERS}
        versions = (version("torch"), version("deepspeed"))
        signatures = {"reduce_ipg_grads": ("self",),
                      "allreduce_and_scatter": ("self", "bucket", "communication_data_type", "numel_per_bucket", "log", "divide", "process_group"),
                      "get_gradient_for_reduction": ("self", "param"),
                      "get_param_comm_dtype": ("self", "param")}
        sig_ok = all(tuple(inspect.signature(getattr(DeepSpeedZeroOptimizer, n)).parameters) == p for n, p in signatures.items())
        ok = versions == (_SUPPORTED_TORCH, _SUPPORTED_DS) and actual == _HELPERS and sig_ok
        return CompatibilityResult(compatible=ok, expected={"torch": _SUPPORTED_TORCH, "deepspeed": _SUPPORTED_DS, "helpers": _HELPERS},
                                   actual={"torch": versions[0], "deepspeed": versions[1], "helpers": actual}, reason="pinned ZeRO-2 helper methods and signatures")
    except Exception as exc:
        return CompatibilityResult(compatible=False, reason=f"ZeRO-2 vendor check failed: {exc}")


def _unsupported_reasons(optimizer):
    errors = []
    for name in ("partition_gradients", "overlap_comm", "contiguous_gradients", "reduce_scatter", "use_multi_rank_bucket_allreduce"):
        if not getattr(optimizer, name, False): errors.append(f"{name} must be enabled")
    for name in ("cpu_offload", "zenflow"):
        if getattr(optimizer, name, False): errors.append(f"{name} is unsupported")
    if getattr(optimizer, "sequence_parallel_size", 1) != 1: errors.append("sequence parallelism unsupported")
    if getattr(optimizer, "gradient_predivide_factor", 1.0) != 1.0: errors.append("gradient predivide unsupported")
    if len(getattr(optimizer, "ipg_buckets", {})) != 1: errors.append("one communication dtype required")
    if getattr(optimizer, "reduction_stream", None) is None: errors.append("async stream required")
    if not dist.is_available() or not dist.is_initialized(): errors.append("distributed process group required")
    elif dist.get_world_size() <= 1 or dist.get_world_size(group=optimizer.dp_process_group) != dist.get_world_size():
        errors.append("data parallel group must cover multiple ranks")
    return errors


def _layout_digest(optimizer):
    digest = hashlib.sha256()
    digest.update(repr(getattr(optimizer, "reduce_bucket_size", None)).encode())
    digest.update(repr(dist.get_world_size(group=optimizer.dp_process_group)).encode())
    for group_index, params in enumerate(optimizer.bit16_groups):
        partitions = optimizer.param_to_partition_ids[group_index]
        offsets = optimizer.grad_start_offset[group_index]
        digest.update(repr(("group", group_index, [p.numel() for p in params])).encode())
        for param_id in sorted(partitions):
            owners = sorted(partitions[param_id])
            digest.update(repr((param_id, owners, [offsets[o][param_id] for o in owners])).encode())
    return digest.hexdigest()


def _runtime_state(optimizer, strict=True):
    state = getattr(optimizer, "_turbo_physai_gro", None)
    if state is not None:
        if not state["activated"] and strict: raise RuntimeError(state["reason"])
        return state["implementation"]
    try:
        errors = _unsupported_reasons(optimizer)
    except Exception as exc:
        errors = [f"runtime validation failed: {exc}"]
    accelerator = None
    if not errors:
        try:
            from deepspeed.accelerator import get_accelerator
            accelerator = get_accelerator()
            if accelerator.resolves_data_dependency():
                errors.append("accelerator dependency semantics unsupported")
        except Exception as exc:
            errors.append(f"accelerator unavailable: {exc}")
    layout = None
    if not errors:
        try:
            layout = _layout_digest(optimizer)
        except Exception as exc:
            errors.append(f"bucket layout validation failed: {exc}")
    status = {"ok": not errors, "reason": "; ".join(errors), "strict": strict, "layout": layout}
    if dist.is_available() and dist.is_initialized():
        all_status = [None] * dist.get_world_size()
        dist.all_gather_object(all_status, status)
        if len({(s["ok"], s["strict"], s["layout"]) for s in all_status}) != 1:
            raise RuntimeError(f"ZeRO-2 rank applicability differs: {all_status}")
    reason = "Unsupported ZeRO-2 overlap runtime: " + status["reason"] if errors else "supported"
    implementation = None
    if not errors:
        implementation = _OverlapImplementation()
        implementation.optimizer = optimizer
        implementation.accelerator = accelerator
        implementation.scheduler = BucketEventScheduler(lambda: implementation.accelerator.current_stream(), torch.cuda.Event)
    optimizer._turbo_physai_gro = {"activated": not errors, "reason": reason, "implementation": implementation, "calls": 0}
    logger.info("PhysAI ZeRO-2 overlap: %s", reason)
    if errors and strict: raise RuntimeError(reason)
    return implementation


def init_wrapper(original, options):
    strict = options.get("strict", True)
    if not isinstance(strict, bool): raise ValueError("strict must be boolean")
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        _runtime_state(self, strict)
        return result
    return wrapped


def _initialized(optimizer):
    state = getattr(optimizer, "_turbo_physai_gro", None)
    if state is None:
        raise RuntimeError("ZeRO-2 overlap was not initialized before backward")
    return state["implementation"]


def reduce_wrapper(original, options):
    strict = options.get("strict", True)
    if not isinstance(strict, bool): raise ValueError("strict must be boolean")
    @wraps(original)
    def wrapped(self, param, i):
        impl = _initialized(self)
        self._turbo_physai_gro["calls"] += 1
        return original(self, param, i) if impl is None else impl.reduce_independent_p_g_buckets_and_remove_grads(param, i)
    return wrapped


def average_wrapper(original, options):
    strict = options.get("strict", True)
    if not isinstance(strict, bool): raise ValueError("strict must be boolean")
    @wraps(original)
    def wrapped(self, tensor, communication_data_type):
        impl = _initialized(self)
        return original(self, tensor, communication_data_type) if impl is None else impl.average_tensor(tensor, communication_data_type)
    return wrapped


def setup_wrapper(original, options):
    @wraps(original)
    def wrapped(self):
        fresh = not self.ready_for_gradients
        result = original(self)
        state = getattr(self, "_turbo_physai_gro", None)
        if fresh and state and state["implementation"] is not None:
            state["implementation"].scheduler.reset()
        return result
    return wrapped
