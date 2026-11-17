# Adapted from Isaac-GR00T, SPDX-License-Identifier: Apache-2.0
"""Experimental ZeRO-2 optimizer-group AllGather pipeline."""
from __future__ import annotations
from copy import copy
from dataclasses import dataclass
from functools import wraps
from importlib.metadata import version
import inspect
import json
import logging
from typing import Any, Callable
import torch
import torch.distributed as dist
from turbo_physai import CompatibilityResult
from turbo_physai.engine.checking.evidence import ast_hash

logger = logging.getLogger(__name__)
_TORCH_VERSION = "2.10.0+das.opt1.dtk2604.2608131652.g28498c"
_DS_VERSION = "0.18.4+das185.dtk2604.torch2100.2608211704.g3ab3df"
_HELPERS = {
 "_optimizer_step": "ast-v1:b61871fca9fdaa362502598398808d1e956b3d19e8c14185ba10fb6e78c2de28",
 "_update_model_bit16_weights": "ast-v1:f43fdc0f6651b0c60ab3a407ba88935b075f15515cbe3a914ed59bebff3b4ada",
}

def _is_fused_adam_type(optimizer_type: type) -> bool:
    try:
        from apex.optimizers import FusedAdam
    except (ImportError, OSError):
        return False
    return optimizer_type is FusedAdam


def _validate_fused_adam_options(optimizer: Any) -> None:
    if (getattr(optimizer, "adam_w_mode", None) != 1
            or getattr(optimizer, "capturable", None) is not False
            or getattr(optimizer, "master_weights", None) is not False):
        raise ValueError("FusedAdam requires AdamW mode without capturable or master weights")


def validate_zero2_optimizer_attributes(
    *,
    stage: int,
    dtype: Any,
    cpu_offload: bool,
    partition_gradients: bool,
    reduce_scatter: bool,
    gradient_accumulation_steps: int,
    optimizer_type: type,
    optimizer_group_count: int,
    zero_group_count: int,
    has_all_gather_into_tensor: bool,
) -> bool:
    import torch

    unsupported = []
    if stage != 2:
        unsupported.append("ZeRO stage must be 2")
    if dtype is not torch.bfloat16:
        unsupported.append("parameter dtype must be BF16")
    if cpu_offload:
        unsupported.append("CPU offload is unsupported")
    if not partition_gradients or not reduce_scatter:
        unsupported.append("ZeRO-2 gradient partitioning and reduce-scatter must be enabled")
    if gradient_accumulation_steps != 1:
        unsupported.append("gradient accumulation steps must be 1")
    if optimizer_type is not torch.optim.AdamW and not _is_fused_adam_type(optimizer_type):
        unsupported.append("optimizer must be torch.optim.AdamW or Apex FusedAdam")
    if optimizer_group_count != zero_group_count:
        unsupported.append("optimizer and ZeRO parameter group counts differ")
    if not has_all_gather_into_tensor:
        unsupported.append("backend must support all_gather_into_tensor")
    if unsupported:
        raise RuntimeError("Unsupported ZeRO-2 AllGather pipeline runtime: " + "; ".join(unsupported))
    return True


def _optimizer_step_for_group(zero_optimizer: Any, index: int) -> None:
    """Align Apex's parallel master list with DeepSpeed's temporary group selection."""
    import torch

    optimizer = zero_optimizer.optimizer
    if type(optimizer) is torch.optim.AdamW:
        zero_optimizer._optimizer_step(index)
        return
    if not _is_fused_adam_type(type(optimizer)):
        raise RuntimeError(f"Unsupported AllGather optimizer: {type(optimizer)}")
    _validate_fused_adam_options(optimizer)
    original_groups = optimizer.param_groups
    original_masters = optimizer.param_groups_master
    group = original_groups[index]
    optimizer.param_groups_master = [{"params": [None] * len(group["params"])}]
    try:
        zero_optimizer._optimizer_step(index)
    finally:
        optimizer.param_groups = original_groups
        optimizer.param_groups_master = original_masters


def _pipeline_zero2_step(self, closure=None):
    """Pinned ZeRO-2 step copy that submits each group AllGather after its update."""
    import torch
    from deepspeed import comm
    from deepspeed.runtime.utils import see_memory_usage
    from deepspeed.runtime.zero.stage_1_and_2 import (
        INITIAL_MICRO_STEP_ID,
        OPTIMIZER_ALLGATHER_TIMER,
        OPTIMIZER_GRADIENTS_TIMER,
        OPTIMIZER_STEP_TIMER,
        OPTIMIZER_TIMERS,
    )

    self.micro_step_id = INITIAL_MICRO_STEP_ID
    see_memory_usage("In step before checking overflow")
    if self.check_grad_overflow:
        self.check_overflow(partition_gradients=self.partition_gradients)

    prev_scale = self.loss_scale
    self._update_scale(self.overflow)
    if self.overflow:
        see_memory_usage("After overflow before clearing gradients")
        self.zero_grad(set_to_none=True)
        for key in self.averaged_gradients:
            self.averaged_gradients[key] = None
            self.all_grad_tensors[key] = None
        see_memory_usage("After overflow after clearing gradients")
        for timer in OPTIMIZER_TIMERS:
            self.timers(timer).start()
            self.timers(timer).stop()
        return

    see_memory_usage("Before norm calculation")
    scaled_global_grad_norm = self.scaled_global_norm()
    self._global_grad_norm = scaled_global_grad_norm / prev_scale
    see_memory_usage("After norm before optimizer")

    pipeline = GroupAllGatherPipeline(lambda index: launch_group_all_gather(self, index, comm))
    self.timers(OPTIMIZER_ALLGATHER_TIMER).start()
    try:
        for index, group in enumerate(self.bit16_groups):
            with torch.profiler.record_function(f"zero2_allgather::optimizer_group_{index}"):
                self.timers(OPTIMIZER_GRADIENTS_TIMER).start()
                partition_id = comm.get_rank(group=self.real_dp_process_group[index])
                self.free_grad_in_param_list(self.params_not_in_partition[index])
                if partition_id == comm.get_world_size(group=self.real_dp_process_group[index]) - 1:
                    single_grad_partition = self.flatten_dense_tensors_aligned(
                        self.averaged_gradients[index], int(self.partition_size[index])
                    ).to(self.single_partition_of_fp32_groups[index].dtype)
                else:
                    single_grad_partition = self.flatten(self.averaged_gradients[index]).to(
                        self.single_partition_of_fp32_groups[index].dtype
                    )
                assert single_grad_partition.numel() == self.partition_size[index], (
                    "averaged gradients have different numbers of elements than partition size "
                    f"{single_grad_partition.numel()} {self.partition_size[index]} {index} {partition_id}"
                )

                self.single_partition_of_fp32_groups[index].grad = single_grad_partition
                self.free_grad_in_param_list(self.params_in_partition[index])
                self.averaged_gradients[index] = None
                self.all_grad_tensors[index] = None
                self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
                self.timers(OPTIMIZER_GRADIENTS_TIMER).stop()

                self.timers(OPTIMIZER_STEP_TIMER).start()
                with torch.profiler.record_function(f"zero2_allgather::adamw_update_{index}"):
                    _optimizer_step_for_group(self, index)
                self.single_partition_of_fp32_groups[index].grad = None
                del single_grad_partition

                bit16_partitions = self.parallel_partitioned_bit16_groups[index]
                fp32_partition = self.single_partition_of_fp32_groups[index]
                with torch.profiler.record_function(f"zero2_allgather::bf16_copy_{index}"):
                    bit16_partitions[partition_id].data.copy_(fp32_partition.data)
                self.timers(OPTIMIZER_STEP_TIMER).stop()

                with torch.profiler.record_function(f"zero2_allgather::submit_{index}"):
                    pipeline.submit(index)

        see_memory_usage("After optimizer before all-gather completion")
        if self.cpu_offload:
            raise RuntimeError("CPU offload became active during the AllGather pipeline step")
        with torch.profiler.record_function("zero2_allgather::wait_all"):
            pipeline.wait_all()
    finally:
        try:
            if pipeline.in_flight_references:
                with torch.profiler.record_function("zero2_allgather::wait_after_error"):
                    _finish_allgathers(self, pipeline)
        finally:
            self.timers(OPTIMIZER_ALLGATHER_TIMER).stop()

    see_memory_usage("After optimizer before weight mapping")
    for index in range(len(self.bit16_groups)):
        with torch.profiler.record_function(f"zero2_allgather::weight_mapping_{index}"):
            self._update_model_bit16_weights(index)
    self.timers.log(OPTIMIZER_TIMERS)
    see_memory_usage("After zero_optimizer step")


def describe_partition_layout(
    group_index: int, parameters: list[Any], flat: Any, partitions: list[Any], rank: int
) -> dict:
    """Validate and describe a ZeRO flat-buffer partition, including padding and aliasing."""
    if not partitions or not 0 <= rank < len(partitions):
        raise RuntimeError(f"Invalid AllGather partition rank for group {group_index}")
    partition_sizes = [part.numel() for part in partitions]
    if len(set(partition_sizes)) != 1:
        raise RuntimeError(f"AllGather partitions must have equal sizes for group {group_index}")
    partition_elements = partition_sizes[rank]
    if flat.numel() != len(partitions) * partition_elements:
        raise RuntimeError(f"AllGather output size mismatch in group {group_index}")
    if flat.dtype != partitions[rank].dtype or flat.device != partitions[rank].device:
        raise RuntimeError(f"AllGather dtype/device mismatch for group {group_index}")
    offset = sum(partition_sizes[:rank])
    expected = flat.narrow(0, offset, partition_elements)
    if partitions[rank].data_ptr() != expected.data_ptr():
        raise RuntimeError(f"AllGather input partition does not alias output in group {group_index}")
    parameter_elements = sum(parameter.numel() for parameter in parameters)
    if parameter_elements > flat.numel():
        raise RuntimeError(f"AllGather flat buffer is smaller than parameters in group {group_index}")
    return {
        "group_index": group_index,
        "parameter_elements": parameter_elements,
        "flat_elements": flat.numel(),
        "padding_elements": flat.numel() - parameter_elements,
        "partition_elements": partition_sizes,
        "local_partition_offset": offset,
        "local_partition_elements": partition_elements,
    }


def regroup_optimizer_parameter_groups(parameter_groups: list[dict], target_elements: int) -> list[dict]:
    """Split optimizer groups by element count without splitting Parameters.

    Source group boundaries are preserved, so parameters with different
    optimizer hyperparameters can never be combined.
    """
    if target_elements <= 0:
        raise ValueError("target_elements must be positive")

    result = []
    for source_index, source in enumerate(parameter_groups):
        parameters = list(source["params"])
        if not parameters:
            raise ValueError(f"Cannot regroup empty optimizer parameter group {source_index}")

        current = []
        current_elements = 0

        def flush():
            nonlocal current, current_elements
            if current:
                group = copy(source)
                group["params"] = current
                result.append(group)
                current = []
                current_elements = 0

        for parameter in parameters:
            parameter_elements = parameter.numel()
            if current and current_elements + parameter_elements > target_elements:
                flush()
            current.append(parameter)
            current_elements += parameter_elements
            if parameter_elements >= target_elements:
                flush()
        flush()
    return result


def regroup_torch_optimizer(optimizer: Any, target_elements: int, parameter_names: dict[int, str]) -> list[dict]:
    """Regroup an unstepped supported AdamW while preserving source-group options."""
    import torch

    fused = type(optimizer) is not torch.optim.AdamW
    if fused:
        if not _is_fused_adam_type(type(optimizer)):
            raise ValueError(f"Expected torch.optim.AdamW or Apex FusedAdam, got {type(optimizer)}")
        _validate_fused_adam_options(optimizer)
    if optimizer.state:
        raise ValueError("Optimizer groups can only be changed before the first optimizer step")

    source_groups = list(optimizer.param_groups)
    regrouped = regroup_optimizer_parameter_groups(source_groups, target_elements)
    manifest = []
    source_index = 0
    sub_index = 0
    for group_index, group in enumerate(regrouped):
        while source_index < len(source_groups) and not any(
            parameter is candidate
            for parameter in group["params"]
            for candidate in source_groups[source_index]["params"]
        ):
            source_index += 1
            sub_index = 0
        if source_index >= len(source_groups):
            raise RuntimeError("Regrouped parameter no longer maps to an original optimizer group")
        names = []
        for parameter in group["params"]:
            name = parameter_names.get(id(parameter))
            if name is None:
                raise ValueError("Every trainable parameter must have a stable model name")
            names.append(name)
        manifest.append(
            {
                "group_index": group_index,
                "source_group_index": source_index,
                "subgroup_index": sub_index,
                "elements": sum(parameter.numel() for parameter in group["params"]),
                "parameter_names": names,
            }
        )
        sub_index += 1

    optimizer.param_groups = regrouped
    if fused:
        optimizer.param_groups_master = [
            {"params": [None] * len(group["params"])} for group in regrouped
        ]
    return manifest


@dataclass(frozen=True)
class PendingAllGather:
    work: Any
    output: Any
    input: Any
    process_group: Any


def launch_group_all_gather(zero_optimizer: Any, group_index: int, comm: Any) -> PendingAllGather | None:
    """Submit one ZeRO parameter group's in-place-alias all-gather asynchronously."""
    process_group = zero_optimizer.real_dp_process_group[group_index]
    world_size = comm.get_world_size(group=process_group)
    if world_size == 1:
        return None

    rank = comm.get_rank(group=process_group)
    output = zero_optimizer.bit16_groups_flat[group_index]
    partitions = zero_optimizer.parallel_partitioned_bit16_groups[group_index]
    if not 0 <= rank < len(partitions) or len(partitions) != world_size:
        raise RuntimeError(f"Invalid ZeRO partition layout for group {group_index}")
    input_tensor = partitions[rank]
    offset = sum(part.numel() for part in partitions[:rank])
    expected_input = output.narrow(0, offset, input_tensor.numel())
    if output.dtype != input_tensor.dtype or output.device != input_tensor.device:
        raise RuntimeError(f"AllGather dtype/device mismatch for group {group_index}")
    if output.numel() != world_size * input_tensor.numel():
        raise RuntimeError(f"AllGather tensor sizes mismatch for group {group_index}")
    if input_tensor.data_ptr() != expected_input.data_ptr():
        raise RuntimeError(f"AllGather input partition must alias its output flat buffer (group {group_index})")

    work = comm.all_gather_into_tensor(
        output_tensor=output,
        tensor=input_tensor,
        group=process_group,
        async_op=True,
    )
    if work is None:
        raise RuntimeError(f"Asynchronous AllGather returned no Work for group {group_index}")
    return PendingAllGather(work, output, input_tensor, process_group)


class GroupAllGatherPipeline:
    """Retain asynchronous gather requests and their tensors until completion."""

    def __init__(self, submit: Callable[[int], Any]):
        self._submit = submit
        self._works = []
        self._references = []

    @property
    def in_flight_references(self):
        return list(self._references)

    def submit(self, group_index: int, tensor_refs: tuple[Any, ...] = ()) -> None:
        result = self._submit(group_index)
        if isinstance(result, PendingAllGather):
            work = result.work
            tensor_refs = (result.output, result.input, result.process_group)
        else:
            work = result
        if work is not None:
            self._works.append(work)
            self._references.append(tensor_refs)

    def wait_all(self) -> None:
        first_error = None
        pending_works = []
        pending_references = []
        for work, references in zip(self._works, self._references):
            try:
                work.wait()
            except BaseException as exc:  # retain storage until failed work is observed again
                pending_works.append(work)
                pending_references.append(references)
                if first_error is None:
                    first_error = exc
        self._works = pending_works
        self._references = pending_references
        if first_error is not None:
            raise first_error


def _finish_allgathers(optimizer: Any, pipeline: GroupAllGatherPipeline) -> None:
    """Keep failed asynchronous requests alive and forbid unsafe reuse of the optimizer."""
    try:
        pipeline.wait_all()
    finally:
        if pipeline.in_flight_references:
            optimizer._turbo_physai_pending_allgather = pipeline
        elif hasattr(optimizer, "_turbo_physai_pending_allgather"):
            del optimizer._turbo_physai_pending_allgather


def allgather_compatibility(context):
    try:
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        actual = {name: ast_hash(getattr(DeepSpeedZeroOptimizer, name)) for name in _HELPERS}
        versions = (version("torch"), version("deepspeed"))
        signatures = {"_optimizer_step": ("self", "group_no"),
                      "_update_model_bit16_weights": ("self", "group_index")}
        actual_signatures = {name: tuple(inspect.signature(getattr(DeepSpeedZeroOptimizer, name)).parameters)
                             for name in signatures}
        ok = versions == (_TORCH_VERSION, _DS_VERSION) and actual == _HELPERS and actual_signatures == signatures
        return CompatibilityResult(compatible=ok, expected={"torch": _TORCH_VERSION, "deepspeed": _DS_VERSION, "helpers": _HELPERS},
                                   actual={"torch": versions[0], "deepspeed": versions[1], "helpers": actual},
                                   reason="pinned AllGather helper implementation and signatures")
    except Exception as exc:
        return CompatibilityResult(compatible=False, reason=f"AllGather compatibility check failed: {exc}")


def scheduler_wrapper(original, options):
    signature = inspect.signature(original)
    target = options.get("group_elements", 100_000_000)
    if type(target) is not int or target <= 0:
        raise ValueError("group_elements must be a positive integer")
    @wraps(original)
    def create(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        optimizer = bound.arguments.get("optimizer") or self.optimizer
        if getattr(self, "is_deepspeed_enabled", False):
            if optimizer is None:
                raise ValueError("AllGather requires an initialized optimizer")
            if not hasattr(optimizer, "_turbo_physai_allgather"):
                if type(optimizer) is torch.optim.AdamW and any(group.get("fused") for group in optimizer.param_groups):
                    raise ValueError("AllGather does not support fused torch AdamW; enable Apex or use adamw_torch")
                names = {id(p): name for name, p in self.model.named_parameters()}
                optimizer._turbo_physai_allgather = regroup_torch_optimizer(optimizer, target, names)
                logger.info("PhysAI GR00T: AllGather regrouped optimizer into %s groups", len(optimizer.param_groups))
        return original(self, *args, **kwargs)
    return create


def train_wrapper(original, options):
    signature = inspect.signature(original)
    @wraps(original)
    def train(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        if bound.arguments.get("resume_from_checkpoint"):
            raise ValueError("AllGather pipeline does not support checkpoint resume")
        a = self.args
        strategy = getattr(getattr(a, "save_strategy", None), "value", getattr(a, "save_strategy", None))
        if strategy != "no" and not (strategy == "steps" and a.save_steps > a.max_steps > 0):
            raise ValueError("AllGather pipeline cannot save checkpoints during training")
        if not a.bf16 or getattr(a, "fp16", False) or a.gradient_accumulation_steps != 1:
            raise ValueError("AllGather pipeline requires BF16 and gradient_accumulation_steps=1")
        if not self.is_deepspeed_enabled:
            raise ValueError("AllGather pipeline requires DeepSpeed ZeRO-2")
        return original(self, *args, **kwargs)
    return train


def _validate_local(optimizer, expected_world):
    from deepspeed import comm
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("initialized distributed group required")
    world = dist.get_world_size()
    if world != expected_world:
        raise RuntimeError(f"AllGather requires {expected_world} ranks, got {world}")
    inner = optimizer.optimizer
    plan = getattr(inner, "_turbo_physai_allgather", None)
    if not isinstance(plan, list) or len(plan) != len(optimizer.bit16_groups):
        raise RuntimeError("AllGather optimizer regrouping plan is missing or incompatible")
    validate_zero2_optimizer_attributes(stage=2 if optimizer.partition_gradients else 1,
        dtype=optimizer.dtype, cpu_offload=optimizer.cpu_offload,
        partition_gradients=optimizer.partition_gradients, reduce_scatter=optimizer.reduce_scatter,
        gradient_accumulation_steps=optimizer.gradient_accumulation_steps, optimizer_type=type(inner),
        optimizer_group_count=len(inner.param_groups), zero_group_count=len(optimizer.bit16_groups),
        has_all_gather_into_tensor=comm.has_all_gather_into_tensor())
    if type(inner) is torch.optim.AdamW:
        if any(group.get("fused") for group in inner.param_groups):
            raise RuntimeError("fused torch AdamW is unsupported")
    else:
        _validate_fused_adam_options(inner)
    if optimizer.torch_autocast_gradscaler or optimizer.zenflow:
        raise RuntimeError("AllGather does not support autocast scaler or ZenFlow")
    layouts = []
    for i, params in enumerate(optimizer.bit16_groups):
        group = optimizer.real_dp_process_group[i]
        if comm.get_world_size(group=group) != world:
            raise RuntimeError("data parallel group must span all ranks")
        flat = optimizer.bit16_groups_flat[i]
        partitions = optimizer.parallel_partitioned_bit16_groups[i]
        layouts.append(describe_partition_layout(i, params, flat, partitions, comm.get_rank(group=group)))
    if [sum(p.numel() for p in group) for group in optimizer.bit16_groups] != [p["elements"] for p in plan]:
        raise RuntimeError("AllGather parameter groups do not match the regrouping plan")
    # Compare optimizer options as well as group names and tensor layouts on
    # every rank before issuing asynchronous collectives.
    options = json.loads(json.dumps(
        [{key: value for key, value in group.items() if key != "params"}
         for group in inner.param_groups], allow_nan=False,
    ))
    return {"plan": plan, "optimizer_options": options,
            "optimizer_class": f"{type(inner).__module__}.{type(inner).__qualname__}",
            "layouts": [{k: v for k, v in item.items() if not k.startswith("local_")} for item in layouts]}


def step_wrapper(original, options):
    strict = options.get("strict", True)
    world = options.get("world_size", 8)
    if not isinstance(strict, bool) or type(world) is not int or world < 2:
        raise ValueError("AllGather requires a boolean strict and world_size >= 2")
    @wraps(original)
    def step(self, closure=None):
        if getattr(self, "_turbo_physai_pending_allgather", None) is not None:
            raise RuntimeError("AllGather optimizer has unfinished asynchronous work")
        if getattr(self, "_turbo_physai_allgather_failed", False):
            raise RuntimeError("AllGather optimizer previously failed during a step and cannot be reused")
        if closure is not None:
            raise ValueError("AllGather pipeline does not support an optimizer closure")
        state = getattr(self, "_turbo_physai_allgather", None)
        if state is None:
            try:
                record = _validate_local(self, world)
                status = {"ok": True, "record": record, "strict": strict, "world_size": world}
            except Exception as exc:
                status = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "strict": strict, "world_size": world}
            statuses = [status]
            if dist.is_available() and dist.is_initialized():
                statuses = [None] * dist.get_world_size()
                dist.all_gather_object(statuses, status)
            if len({(item["strict"], item["world_size"]) for item in statuses}) != 1:
                raise RuntimeError("AllGather rank options disagree")
            active = all(item["ok"] for item in statuses)
            if active and any(item["record"] != statuses[0]["record"] for item in statuses[1:]):
                raise RuntimeError("AllGather rank layouts or plans disagree")
            state = {"activated": active, "statuses": statuses, "calls": 0}
            self._turbo_physai_allgather = state
            logger.info("PhysAI GR00T AllGather pipeline: %s", "active" if active else statuses)
            if not active and strict:
                raise RuntimeError(f"AllGather runtime validation failed: {statuses}")
        if not state["activated"]:
            if strict:
                raise RuntimeError(f"AllGather runtime validation failed: {state['statuses']}")
            return original(self, closure=closure)
        state["calls"] += 1
        try:
            return _pipeline_zero2_step(self, closure=closure)
        except BaseException:
            self._turbo_physai_allgather_failed = True
            raise
    return step
