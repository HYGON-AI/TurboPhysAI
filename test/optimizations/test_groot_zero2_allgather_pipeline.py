"""ZeRO-2 AllGather pipeline, optimizer regrouping and failure cleanup."""

from copy import deepcopy
from datetime import timedelta
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from turbo_physai.optimizations.models.groot.zero2_allgather import (
    GroupAllGatherPipeline,
    describe_partition_layout,
    launch_group_all_gather,
    load_state_dict_wrapper,
    regroup_optimizer_parameter_groups,
    regroup_torch_optimizer,
    scheduler_wrapper,
    state_dict_wrapper,
    step_wrapper,
    train_wrapper,
)


# Zero2 allgather pipeline

def _trainer(optimizer):
    params = [p for group in optimizer.param_groups for p in group["params"]]
    return SimpleNamespace(
        is_deepspeed_enabled=True,
        optimizer=optimizer,
        model=SimpleNamespace(named_parameters=lambda: [(f"p{i}", p) for i, p in enumerate(params)]),
        args=SimpleNamespace(bf16=True, fp16=False, gradient_accumulation_steps=1,
                             save_strategy="steps", save_steps=1000, max_steps=10),
    )


def test_scheduler_tracks_all_regrouped_parameters():
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    optimizer = torch.optim.AdamW(params, lr=0.01, fused=False)
    trainer = _trainer(optimizer)
    def create(self, steps, optimizer=None):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    scheduler = scheduler_wrapper(create, {"group_elements": 2})(trainer, 10, optimizer=optimizer)
    assert len(optimizer.param_groups) == len(scheduler.base_lrs) == 3
    optimizer.step()
    scheduler.step()
    assert [g["lr"] for g in optimizer.param_groups] == [0.005] * 3
    assert len(optimizer._turbo_physai_allgather) == 3


def test_fused_torch_adamw_is_rejected():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], fused=True)
    trainer = _trainer(optimizer)
    with pytest.raises(ValueError, match="fused"):
        scheduler_wrapper(lambda *args, **kwargs: None, {})(trainer, 10, optimizer=optimizer)


def test_apex_selection_precedes_scheduler_regroup(monkeypatch):
    from turbo_physai.optimizations.models.groot import zero2_allgather as module
    from turbo_physai.optimizations.models.groot.optim import create_optimizer_apex_wrapper

    class FakeFusedAdam(torch.optim.AdamW):
        adam_w_mode = 1
        capturable = False
        master_weights = False

    monkeypatch.setattr(module, "_is_fused_adam_type", lambda kind: kind is FakeFusedAdam)
    params = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    trainer = _trainer(torch.optim.AdamW(params))
    trainer.args.optim = "adamw_torch_fused"
    events = []

    def create_optimizer(self):
        events.append(self.args.optim)
        self.optimizer = FakeFusedAdam(params, lr=0.01)
        return self.optimizer

    def create_scheduler(self, steps, optimizer=None):
        events.append(len(optimizer.param_groups))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)

    optimizer = create_optimizer_apex_wrapper(create_optimizer, {})(trainer)
    scheduler = scheduler_wrapper(create_scheduler, {"group_elements": 2})(trainer, 10, optimizer=optimizer)
    assert events == ["adamw_apex_fused", 2]
    assert trainer.args.optim == "adamw_torch_fused"
    assert len(scheduler.base_lrs) == len(optimizer.param_groups_master) == 2


@pytest.mark.parametrize("keyword", [{"resume_from_checkpoint": "x"}])
def test_train_rejects_resume(keyword):
    trainer = _trainer(torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    def original(self, resume_from_checkpoint=None):
        return "started"
    with pytest.raises(ValueError, match="resume"):
        train_wrapper(original, {})(trainer, **keyword)


def test_train_can_resume_with_periodic_saves_when_enabled():
    trainer = _trainer(torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    trainer.args.save_steps = 2

    def original(self, resume_from_checkpoint=None):
        return resume_from_checkpoint

    train = train_wrapper(original, {"allow_checkpointing": True})
    assert train(trainer, resume_from_checkpoint="checkpoint-2") == "checkpoint-2"


@pytest.fixture
def zero_checkpoint(monkeypatch):
    pytest.importorskip("deepspeed")
    from deepspeed import comm
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.utils.tensor_fragment import fragment_address

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(comm, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(comm, "has_all_gather_into_tensor", lambda: True)

    def gather(output, value):
        output[:] = [value] * 2

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    def create(rank):
        monkeypatch.setattr(comm, "get_rank", lambda group=None: rank)
        master = torch.nn.Parameter(torch.full((2,), float(rank + 1)))
        optimizer = torch.optim.AdamW([master], lr=0.01)
        optimizer._turbo_physai_allgather = [{
            "group_index": 0, "source_group_index": 0, "subgroup_index": 0,
            "elements": 4, "parameter_names": ["layer.weight"],
        }]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / 4)
        flat = torch.zeros(4, dtype=torch.bfloat16)
        zero = object.__new__(DeepSpeedZeroOptimizer)
        zero.__dict__.update(
            optimizer=optimizer, dtype=torch.bfloat16, cpu_offload=False, partition_gradients=True,
            reduce_scatter=True, gradient_accumulation_steps=1, torch_autocast_gradscaler=False,
            zenflow=False, dp_process_group=None, real_dp_process_group=[None],
            bit16_groups=[[flat]], bit16_groups_flat=[flat],
            parallel_partitioned_bit16_groups=[list(flat.chunk(2))],
            loss_scaler=None, dynamic_loss_scale=False, overflow=False, clip_grad=1.0,
            elastic_checkpoint=False, single_partition_of_fp32_groups=[master],
            groups_padding=[0], partition_count=[2],
            _param_slice_mappings=[{"layer.weight": fragment_address(start=0, numel=2)}],
            _link_all_hp_params=lambda: None,
        )
        options = {"world_size": 2, "allow_checkpointing": True}
        save = state_dict_wrapper(DeepSpeedZeroOptimizer.state_dict, options)
        load = load_state_dict_wrapper(DeepSpeedZeroOptimizer.load_state_dict, options)
        return zero, master, scheduler, save, load

    return create


@pytest.mark.model_deps
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("sparse_shards", [True, False])
def test_allgather_checkpoint_restores_native_optimizer_and_lr(zero_checkpoint, tmp_path, rank, sparse_shards):
    zero, master, scheduler, save, _ = zero_checkpoint(rank)

    def update(zero, master, scheduler, gradient):
        master.grad = torch.full_like(master, gradient)
        zero.optimizer.step()
        scheduler.step()

    update(zero, master, scheduler, 0.1)
    update(zero, master, scheduler, -0.2)
    checkpoint_path = tmp_path / "optimizer.pt"
    torch.save(save(zero), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    scheduler_state = deepcopy(scheduler.state_dict())
    restored, restored_master, restored_scheduler, _, load = zero_checkpoint(rank)
    assert restored.optimizer.param_groups[0]["lr"] != zero.optimizer.param_groups[0]["lr"]
    shards = [None, None] if sparse_shards else [checkpoint, checkpoint]
    shards[rank] = checkpoint
    load(restored, shards, load_from_fp32_weights=True)
    restored_scheduler.load_state_dict(scheduler_state)
    torch.testing.assert_close(restored_master, master, rtol=0, atol=0)
    assert restored.optimizer.param_groups[0]["lr"] == zero.optimizer.param_groups[0]["lr"]

    update(zero, master, scheduler, 0.3)
    update(restored, restored_master, restored_scheduler, 0.3)
    torch.testing.assert_close(restored_master, master, rtol=0, atol=0)
    torch.testing.assert_close(
        restored.optimizer.state[restored_master], zero.optimizer.state[master], rtol=0, atol=0,
    )
    assert restored_scheduler.get_last_lr() == scheduler.get_last_lr()


@pytest.mark.model_deps
def test_allgather_checkpoint_rejects_changed_rank_local_parameter_slices(zero_checkpoint):
    original, _, _, save, _ = zero_checkpoint(0)
    checkpoint = save(original)
    restored, _, _, _, load = zero_checkpoint(0)
    restored._param_slice_mappings[0]["layer.weight"].start = 1
    with pytest.raises(RuntimeError, match="layout"):
        load(restored, [checkpoint, None], load_from_fp32_weights=True)
    assert not restored.optimizer.state


@pytest.mark.model_deps
def test_rank_count_fallback_rejects_pipelined_checkpoint(zero_checkpoint):
    original, _, _, save, _ = zero_checkpoint(0)
    checkpoint = save(original)
    restored, _, _, _, _ = zero_checkpoint(0)

    def native_load(*args, **kwargs):
        pytest.fail("native load was called with an AllGather checkpoint")

    load = load_state_dict_wrapper(native_load, {"world_size": 8, "strict": False})
    with pytest.raises(RuntimeError, match="Cannot load an AllGather checkpoint"):
        load(restored, [checkpoint, None])


def test_allgather_checkpoint_rejects_pending_work_before_native_save():
    optimizer = SimpleNamespace(_turbo_physai_allgather=[])
    zero = SimpleNamespace(optimizer=optimizer, _turbo_physai_pending_allgather=object())

    def native_save(self):
        pytest.fail("native save was called with pending AllGather work")

    with pytest.raises(RuntimeError, match="unfinished"):
        state_dict_wrapper(native_save, {})(zero)


def _exercise_rank_mismatch_checkpoint(rank, directory, phase):
    import os

    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    from turbo_physai.optimizations.models.groot.zero2_allgather import _MANIFEST_KEY

    torch.set_num_threads(1)
    os.environ.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE="2")
    directory = Path(directory)
    torch.distributed.init_process_group(
        "gloo", init_method=(directory / f"{phase}.store").as_uri(),
        rank=rank, world_size=2, timeout=timedelta(seconds=45),
    )
    # Use Gloo without building DeepSpeed's optional CPU shared-memory extension.
    import_module("deepspeed.comm.torch").build_shm_op = lambda: None
    deepspeed.init_distributed("gloo", auto_mpi_discovery=False, dist_init_required=False)

    options = dict(world_size=8, strict=False, allow_checkpointing=True, group_elements=64)

    def create(fallback):
        torch.manual_seed(12)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, fused=False)

        def create_scheduler(self, optimizer):
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)

        scheduler = scheduler_wrapper(create_scheduler, options) if fallback else create_scheduler
        engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer, lr_scheduler=scheduler(_trainer(optimizer), optimizer),
            config={
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "bf16": {"enabled": True},
                "zero_allow_untested_optimizer": True,
                "zero_optimization": {"stage": 2, "reduce_scatter": True, "overlap_comm": False},
            },
        )
        return engine

    def update(engine, step):
        generator = torch.Generator().manual_seed(100 + 10 * rank + step)
        batch = torch.randn(1, 8, generator=generator).bfloat16()
        engine.backward(engine(batch).float().square().mean())
        engine.step()

    def snapshot(engine):
        return {
            "model": engine.module.state_dict(),
            "master_weights": engine.optimizer.single_partition_of_fp32_groups,
            "adamw": engine.optimizer.optimizer.state_dict(),
            "scheduler": engine.lr_scheduler.state_dict(),
            "steps": engine.global_steps,
        }

    reference = directory / f"reference-{rank}.pt"
    checkpoint = str(directory / "checkpoint")
    try:
        if phase == "save":
            baseline = create(fallback=False)
            update(baseline, 0)
            update(baseline, 1)
            torch.save(snapshot(baseline), reference)

        DeepSpeedZeroOptimizer.step = step_wrapper(DeepSpeedZeroOptimizer.step, options)
        DeepSpeedZeroOptimizer.state_dict = state_dict_wrapper(DeepSpeedZeroOptimizer.state_dict, options)
        DeepSpeedZeroOptimizer.load_state_dict = load_state_dict_wrapper(DeepSpeedZeroOptimizer.load_state_dict, options)
        engine = create(fallback=True)

        if phase == "save":
            update(engine, 0)
            assert engine.save_checkpoint(checkpoint, tag="step-1")
            assert _MANIFEST_KEY not in engine.optimizer.state_dict()
        else:
            loaded, _ = engine.load_checkpoint(checkpoint, tag="step-1")
            assert loaded and engine.global_steps == 1
            update(engine, 1)
            expected = torch.load(reference, map_location="cpu", weights_only=False)
            torch.testing.assert_close(snapshot(engine), expected, rtol=0, atol=0)
        assert len(engine.optimizer.optimizer.param_groups) == 1
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.model_deps
def test_rank_mismatch_falls_back_and_resumes_native_checkpoint(tmp_path, monkeypatch):
    if find_spec("deepspeed") is None or not torch.distributed.is_gloo_available():
        pytest.skip("requires DeepSpeed and Gloo")
    monkeypatch.setenv("DS_ACCELERATOR", "cpu")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    for phase in ("save", "resume"):
        torch.multiprocessing.spawn(
            _exercise_rank_mismatch_checkpoint, args=(str(tmp_path), phase), nprocs=2,
        )


def test_pipeline_waits_for_all_work_even_when_one_fails():
    calls = []
    class Work:
        def __init__(self, index): self.index = index
        def wait(self):
            calls.append(self.index)
            if self.index == 0: raise RuntimeError("failed")
    pipeline = GroupAllGatherPipeline(lambda i: Work(i))
    pipeline.submit(0, (object(),))
    pipeline.submit(1, (object(),))
    with pytest.raises(RuntimeError, match="failed"):
        pipeline.wait_all()
    assert calls == [0, 1]
    assert len(pipeline.in_flight_references) == 1


def test_failed_async_work_is_retained_and_prevents_another_step():
    from turbo_physai.optimizations.models.groot.zero2_allgather import _finish_allgathers

    class Work:
        def wait(self):
            raise RuntimeError("communication failed")

    pipeline = GroupAllGatherPipeline(lambda _: Work())
    references = (object(),)
    pipeline.submit(0, references)
    optimizer = SimpleNamespace()
    with pytest.raises(RuntimeError, match="communication failed"):
        _finish_allgathers(optimizer, pipeline)
    assert optimizer._turbo_physai_pending_allgather is pipeline
    assert pipeline.in_flight_references == [references]
    with pytest.raises(RuntimeError, match="unfinished"):
        step_wrapper(lambda self, closure=None: None, {})(optimizer)


def test_failed_partial_step_permanently_poisoned(monkeypatch):
    from turbo_physai.optimizations.models.groot import zero2_allgather as module

    def fail_after_update(optimizer, closure=None):
        optimizer.updates += 1
        raise RuntimeError("group update failed")

    monkeypatch.setattr(module, "_pipeline_zero2_step", fail_after_update)
    optimizer = SimpleNamespace(updates=0, _turbo_physai_allgather={"activated": True, "calls": 0})
    wrapped = step_wrapper(lambda self, closure=None: None, {})
    with pytest.raises(RuntimeError, match="group update failed"):
        wrapped(optimizer)
    with pytest.raises(RuntimeError, match="previously failed"):
        wrapped(optimizer)
    assert optimizer.updates == 1


@pytest.mark.parametrize("settings", [
    {"bf16": False}, {"gradient_accumulation_steps": 2},
    {"save_strategy": "steps", "save_steps": 2, "max_steps": 10},
])
def test_train_rejects_unsupported_configuration(settings):
    trainer = _trainer(torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    for key, value in settings.items():
        setattr(trainer.args, key, value)
    with pytest.raises(ValueError):
        train_wrapper(lambda self, resume_from_checkpoint=None: None, {})(trainer)


def test_step_rejects_rank_plan_mismatch(monkeypatch):
    from turbo_physai.optimizations.models.groot import zero2_allgather as module
    monkeypatch.setattr(module, "_validate_local", lambda obj, world: {"plan": ["rank-0"]})
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    def gather(items, status):
        items[:] = [status, {**status, "record": {"plan": ["rank-1"]}}]
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    wrapped = step_wrapper(lambda self, closure=None: None, {"world_size": 2})
    with pytest.raises(RuntimeError, match="disagree"):
        wrapped(SimpleNamespace())


def test_step_rejects_different_strict_options_across_ranks(monkeypatch):
    from turbo_physai.optimizations.models.groot import zero2_allgather as module
    monkeypatch.setattr(module, "_validate_local", lambda obj, world: {"plan": []})
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    def gather(items, status):
        items[:] = [status, {**status, "strict": False}]
    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    with pytest.raises(RuntimeError, match="options disagree"):
        step_wrapper(lambda self, closure=None: None, {"world_size": 2})(SimpleNamespace())


def test_step_strict_false_falls_back_on_validation_failure(monkeypatch):
    from turbo_physai.optimizations.models.groot import zero2_allgather as module
    def invalid(*args): raise RuntimeError("unsupported")
    monkeypatch.setattr(module, "_validate_local", invalid)
    monkeypatch.setattr(module.dist, "is_available", lambda: False)
    original = lambda self, closure=None: "fallback"
    optimizer = SimpleNamespace()
    wrapped = step_wrapper(original, {"strict": False})
    assert wrapped(optimizer) == "fallback"
    assert optimizer._turbo_physai_allgather["activated"] is False


# ZeRO-2 AllGather pipeline reference tests.

def test_pipeline_overflow_clears_gradients_without_updating_or_gathering():
    from types import SimpleNamespace

    from turbo_physai.optimizations.models.groot.zero2_allgather import _pipeline_zero2_step

    events = []
    timer = SimpleNamespace(start=lambda: events.append("timer_start"),
                            stop=lambda: events.append("timer_stop"))
    optimizer = SimpleNamespace(
        micro_step_id=9, check_grad_overflow=True, partition_gradients=True,
        loss_scale=2.0, overflow=True,
        check_overflow=lambda **kwargs: events.append(("check_overflow", kwargs)),
        _update_scale=lambda overflow: events.append(("update_scale", overflow)),
        zero_grad=lambda **kwargs: events.append(("zero_grad", kwargs)),
        averaged_gradients={0: object(), 1: object()},
        all_grad_tensors={0: object(), 1: object()},
        timers=lambda _: timer,
    )

    _pipeline_zero2_step(optimizer)

    assert optimizer.micro_step_id != 9
    assert optimizer.averaged_gradients == {0: None, 1: None}
    assert optimizer.all_grad_tensors == {0: None, 1: None}
    assert events[:3] == [
        ("check_overflow", {"partition_gradients": True}),
        ("update_scale", True),
        ("zero_grad", {"set_to_none": True}),
    ]
    assert events.count("timer_start") == events.count("timer_stop")


def test_regroup_keeps_parameters_whole_ordered_and_hyperparameters():
    import torch

    first = [torch.nn.Parameter(torch.zeros(size)) for size in (6, 5, 2, 12)]
    second = [torch.nn.Parameter(torch.zeros(size)) for size in (4, 4)]
    groups = [
        {"params": first, "weight_decay": 0.1, "lr": 0.01, "group_name": "decay"},
        {"params": second, "weight_decay": 0.0, "lr": 0.01, "group_name": "no_decay"},
    ]

    regrouped = regroup_optimizer_parameter_groups(groups, target_elements=10)

    assert [len(group["params"]) for group in regrouped] == [1, 2, 1, 2]
    assert [param for group in regrouped for param in group["params"]] == first + second
    assert [group["group_name"] for group in regrouped] == ["decay"] * 3 + ["no_decay"]
    assert [group["weight_decay"] for group in regrouped] == [0.1] * 3 + [0.0]
    assert sum(param.numel() for group in groups for param in group["params"]) == sum(
        param.numel() for group in regrouped for param in group["params"]
    )
    assert len(groups) == 2


def test_regroup_rejects_empty_parameter_groups():
    import torch

    groups = [{"params": [torch.nn.Parameter(torch.ones(1))]}, {"params": []}]

    try:
        regroup_optimizer_parameter_groups(groups, target_elements=10)
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("empty source parameter groups must be rejected")


def test_pipeline_waits_for_all_gathers_after_all_groups_are_submitted():
    events = []

    class Work:
        def __init__(self, group_index):
            self.group_index = group_index

        def wait(self):
            events.append(("wait", self.group_index))

    pipeline = GroupAllGatherPipeline(lambda group_index: (events.append(("submit", group_index)) or Work(group_index)))
    for group_index in range(3):
        events.append(("update", group_index))
        events.append(("copy", group_index))
        pipeline.submit(group_index)

    pipeline.wait_all()
    events.append(("map_weights", None))

    assert events == [
        ("update", 0),
        ("copy", 0),
        ("submit", 0),
        ("update", 1),
        ("copy", 1),
        ("submit", 1),
        ("update", 2),
        ("copy", 2),
        ("submit", 2),
        ("wait", 0),
        ("wait", 1),
        ("wait", 2),
        ("map_weights", None),
    ]


def test_pipeline_retains_tensors_until_work_completes():
    released = []

    class Work:
        def wait(self):
            released.append("wait")

    pipeline = GroupAllGatherPipeline(lambda _: (released.append("submit") or Work()))
    tensor_refs = (object(), object())
    pipeline.submit(0, tensor_refs=tensor_refs)

    assert pipeline.in_flight_references == [tensor_refs]
    pipeline.wait_all()
    assert released == ["submit", "wait"]
    assert pipeline.in_flight_references == []


def test_pipeline_keeps_failed_work_references_for_error_cleanup():
    class Work:
        def wait(self):
            raise RuntimeError("device communication failed")

    pipeline = GroupAllGatherPipeline(lambda _: Work())
    refs = (object(), object())
    pipeline.submit(0, tensor_refs=refs)

    try:
        pipeline.wait_all()
    except RuntimeError as exc:
        assert "communication failed" in str(exc)
    else:
        raise AssertionError("failed work must propagate its error")

    assert pipeline.in_flight_references == [refs]


def test_group_gather_uses_rank_partition_alias_and_async_backend():
    import torch

    flat = torch.arange(8, dtype=torch.bfloat16)
    partitions = [flat.narrow(0, 0, 4), flat.narrow(0, 4, 4)]
    calls = []
    expected_work = object()

    class Comm:
        @staticmethod
        def get_world_size(group):
            assert group == "dp"
            return 2

        @staticmethod
        def get_rank(group):
            assert group == "dp"
            return 1

        @staticmethod
        def all_gather_into_tensor(**kwargs):
            calls.append(kwargs)
            return expected_work

    zero = type(
        "Zero",
        (),
        {
            "real_dp_process_group": ["dp"],
            "bit16_groups_flat": [flat],
            "parallel_partitioned_bit16_groups": [partitions],
        },
    )()

    request = launch_group_all_gather(zero, 0, Comm)

    assert request.work is expected_work
    assert request.output is flat
    assert request.input is partitions[1]
    assert request.input.data_ptr() == flat.narrow(0, 4, 4).data_ptr()
    assert calls == [{"output_tensor": flat, "tensor": partitions[1], "group": "dp", "async_op": True}]


def test_group_gather_rejects_nonaliased_input_partition():
    import torch

    class Comm:
        @staticmethod
        def get_world_size(group):
            return 2

        @staticmethod
        def get_rank(group):
            return 0

    zero = type(
        "Zero",
        (),
        {
            "real_dp_process_group": [None],
            "bit16_groups_flat": [torch.zeros(8, dtype=torch.bfloat16)],
            "parallel_partitioned_bit16_groups": [
                [torch.zeros(4, dtype=torch.bfloat16), torch.zeros(4, dtype=torch.bfloat16)]
            ],
        },
    )()

    try:
        launch_group_all_gather(zero, 0, Comm)
    except RuntimeError as exc:
        assert "alias" in str(exc)
    else:
        raise AssertionError("nonaliased gather storage must fail closed")


def test_partition_layout_records_offsets_padding_and_checks_equal_parts():
    import torch

    flat = torch.zeros(12, dtype=torch.bfloat16)
    parts = [flat.narrow(0, 0, 4), flat.narrow(0, 4, 4), flat.narrow(0, 8, 4)]
    params = [torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.zeros(6))]

    layout = describe_partition_layout(2, params, flat, parts, rank=1)

    assert layout == {
        "group_index": 2,
        "parameter_elements": 11,
        "flat_elements": 12,
        "padding_elements": 1,
        "partition_elements": [4, 4, 4],
        "local_partition_offset": 4,
        "local_partition_elements": 4,
    }

    unequal = [flat.narrow(0, 0, 3), flat.narrow(0, 3, 5), flat.narrow(0, 8, 4)]
    try:
        describe_partition_layout(2, params, flat, unequal, rank=1)
    except RuntimeError as exc:
        assert "equal" in str(exc)
    else:
        raise AssertionError("AllGather must reject unequal partition sizes")


def test_torch_optimizer_regroup_preserves_decay_groups_names_and_parameter_order():
    import torch

    decay = [torch.nn.Parameter(torch.ones(6)), torch.nn.Parameter(torch.ones(3))]
    no_decay = [torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.ones(2))]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1, "lr": 1e-4},
            {"params": no_decay, "weight_decay": 0.0, "lr": 1e-4},
        ],
        eps=1e-8,
    )
    names = {id(parameter): f"p{index}" for index, parameter in enumerate(decay + no_decay)}

    manifest = regroup_torch_optimizer(optimizer, target_elements=8, parameter_names=names)

    assert [parameter for group in optimizer.param_groups for parameter in group["params"]] == decay + no_decay
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.1, 0.0]
    assert [entry["parameter_names"] for entry in manifest] == [["p0"], ["p1"], ["p2", "p3"]]
    assert all(group["eps"] == 1e-8 and group["lr"] == 1e-4 for group in optimizer.param_groups)


def test_torch_optimizer_regroup_rejects_existing_state():
    import torch

    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.state[parameter]["step"] = torch.tensor(1.0)

    try:
        regroup_torch_optimizer(optimizer, target_elements=1, parameter_names={id(parameter): "p"})
    except ValueError as exc:
        assert "before the first optimizer step" in str(exc)
    else:
        raise AssertionError("optimizer groups cannot change after state creation")


@pytest.fixture
def apex_fused_adam():
    if find_spec("apex") is None:
        pytest.skip("requires the DTK Apex FusedAdam build")
    from apex.optimizers import fused_adam

    return fused_adam


@pytest.mark.model_deps
def test_fused_optimizer_group_step_matches_full_step_and_restores_lists(monkeypatch, apex_fused_adam):
    """Exercise real Apex Python step with its GPU kernel replaced by CPU math."""
    import torch
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from turbo_physai.optimizations.models.groot.zero2_allgather import _optimizer_step_for_group

    def cpu_apply(_kernel, _overflow, tensors, lr, beta1, beta2, eps, step,
                  adam_w_mode, bias_correction, weight_decay):
        assert adam_w_mode == bias_correction == 1
        with torch.no_grad():
            for grad, param, mean, variance in zip(*tensors):
                mean.mul_(beta1).add_(grad, alpha=1 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = variance.sqrt().div_((1 - beta2 ** step) ** .5).add_(eps)
                param.mul_(1 - lr * weight_decay)
                param.addcdiv_(mean, denom, value=-lr / (1 - beta1 ** step))

    monkeypatch.setattr(apex_fused_adam, "multi_tensor_applier", cpu_apply)

    def make_optimizer():
        parameters = [torch.nn.Parameter(torch.tensor([1.0, 2.0])),
                      torch.nn.Parameter(torch.tensor([3.0, 4.0, 5.0])),
                      torch.nn.Parameter(torch.tensor([6.0]))]
        optimizer = apex_fused_adam.FusedAdam.__new__(apex_fused_adam.FusedAdam)
        torch.optim.Optimizer.__init__(optimizer, [{"params": [p]} for p in parameters],
            {"lr": .01, "betas": (.9, .99), "eps": 1e-8,
             "weight_decay": .01, "bias_correction": True})
        optimizer.param_groups_master = [{"params": [None]} for _ in parameters]
        optimizer.adam_w_mode, optimizer.capturable, optimizer.master_weights = 1, False, False
        optimizer._dummy_overflow_buf = torch.zeros(1, dtype=torch.int)
        optimizer.multi_tensor_adam = object()
        return optimizer, parameters

    reference, expected = make_optimizer()
    actual, observed = make_optimizer()
    # Mimic regrouping having invalidated Apex's parallel group list.
    actual.param_groups_master = [{"params": []}]
    stale_master = actual.param_groups_master
    original_groups = actual.param_groups

    class ZeroOptimizer:
        optimizer = actual
        torch_autocast_gradscaler = False
        zenflow = False

        def _lazy_init_hp_params_optimizer_state(self):
            pass

        def _optimizer_step(self, index):
            return DeepSpeedZeroOptimizer._optimizer_step(self, index)

    for step in range(3):
        for index, (left, right) in enumerate(zip(expected, observed)):
            left.grad = torch.full_like(left, (step + 1) * (index + 1) / 10)
            right.grad = left.grad.clone()
        reference.step()
        for index in range(len(observed)):
            _optimizer_step_for_group(ZeroOptimizer(), index)
        for left, right in zip(expected, observed):
            torch.testing.assert_close(left, right)
            torch.testing.assert_close(reference.state[left]["exp_avg"], actual.state[right]["exp_avg"])
            torch.testing.assert_close(reference.state[left]["exp_avg_sq"], actual.state[right]["exp_avg_sq"])
        assert actual.param_groups is original_groups
        assert actual.param_groups_master is stale_master
    assert [g['step'] for g in actual.param_groups] == [3, 3, 3]


@pytest.mark.model_deps
def test_fused_group_adapter_restores_lists_if_step_raises(apex_fused_adam):
    from turbo_physai.optimizations.models.groot.zero2_allgather import _optimizer_step_for_group

    param = torch.nn.Parameter(torch.ones(1))
    optimizer = apex_fused_adam.FusedAdam.__new__(apex_fused_adam.FusedAdam)
    torch.optim.Optimizer.__init__(optimizer, [param], {})
    optimizer.adam_w_mode, optimizer.capturable, optimizer.master_weights = 1, False, False
    optimizer.param_groups_master = [{"params": [None]}]
    groups, masters = optimizer.param_groups, optimizer.param_groups_master

    class ZeroOptimizer:
        def __init__(self): self.optimizer = optimizer
        def _optimizer_step(self, index):
            self.optimizer.param_groups = [groups[index]]
            raise RuntimeError("update failed")

    with pytest.raises(RuntimeError, match="update failed"):
        _optimizer_step_for_group(ZeroOptimizer(), 0)
    assert optimizer.param_groups is groups
    assert optimizer.param_groups_master is masters


def test_plain_adamw_group_step_does_not_import_apex(monkeypatch):
    import builtins
    import torch
    from turbo_physai.optimizations.models.groot.zero2_allgather import _optimizer_step_for_group

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    events = []
    class ZeroOptimizer:
        def __init__(self): self.optimizer = optimizer
        def _optimizer_step(self, index): events.append(index)
    original_import = builtins.__import__
    def import_without_apex(name, *args, **kwargs):
        if name.startswith('apex'):
            raise AssertionError('ordinary AdamW must not import Apex')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', import_without_apex)
    _optimizer_step_for_group(ZeroOptimizer(), 0)
    assert events == [0]
