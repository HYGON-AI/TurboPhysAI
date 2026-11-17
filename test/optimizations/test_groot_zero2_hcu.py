"""Launch with torchrun on two/eight HCUs; these tests exercise real ZeRO-2 engines."""

import os

import pytest
import torch


pytestmark = [pytest.mark.hcu, pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip or "RANK" not in os.environ,
    reason="requires torchrun and multiple HCUs",
)]


def _engines(monkeypatch, *, owner=False, allgather=False):
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as Zero
    from turbo_physai.optimizations.models.groot import zero2_overlap as gro, zero2_owner_reduce as obr, zero2_allgather as ag

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not torch.distributed.is_initialized():
        deepspeed.init_distributed()
    assert world in (2, 8)
    for name, factory in (("__init__", gro.init_wrapper),
                          ("reduce_independent_p_g_buckets_and_remove_grads", gro.reduce_wrapper),
                          ("average_tensor", gro.average_wrapper), ("setup_buckets", gro.setup_wrapper)):
        monkeypatch.setattr(Zero, name, factory(getattr(Zero, name), {}))
    config = {"train_micro_batch_size_per_gpu": 1, "gradient_accumulation_steps": 1,
              "bf16": {"enabled": True}, "zero_allow_untested_optimizer": True,
              "zero_optimization": {"stage": 2, "overlap_comm": True, "reduce_scatter": True,
                                    "contiguous_gradients": True, "reduce_bucket_size": 8192,
                                    "allgather_bucket_size": 256}}
    def create():
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Linear(32, 32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)
        if allgather:
            optimizer._turbo_physai_allgather = ag.regroup_torch_optimizer(
                optimizer, 1000, {id(param): name for name, param in model.named_parameters()})
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
        return engine

    reference, actual = create(), create()
    if owner:
        original = Zero.allreduce_and_scatter
        wrapped = obr.owner_reduce_wrapper(original, {})
    elif allgather:
        original = Zero.step
        wrapped = ag.step_wrapper(original, {"world_size": world})
    else:
        raise AssertionError("one optimization required")
    return rank, world, Zero, reference, actual, original, wrapped


@pytest.mark.parametrize("kind", ["owner", "allgather"])
def test_zero2_optimized_matches_reference(monkeypatch, kind):
    rank, world, zero_cls, reference, actual, original, wrapped = _engines(
        monkeypatch, owner=kind == "owner", allgather=kind == "allgather")
    if kind == "allgather" and world != 8:
        pytest.skip("AllGather integration gate requires eight ranks")
    name = "allreduce_and_scatter" if kind == "owner" else "step"
    for step in range(3):
        torch.manual_seed(100 + step + rank * 100)
        inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        monkeypatch.setattr(zero_cls, name, original)
        reference.backward(reference(inputs).float().square().mean())
        monkeypatch.setattr(zero_cls, name, wrapped)
        actual.backward(actual(inputs).float().square().mean())
        torch.cuda.synchronize()
        compared_grads = 0
        for index in reference.optimizer.averaged_gradients:
            expected_grads = reference.optimizer.averaged_gradients[index]
            result_grads = actual.optimizer.averaged_gradients[index]
            assert (expected_grads is None) == (result_grads is None)
            if expected_grads is not None:
                assert len(expected_grads) == len(result_grads)
                for expected_grad, result_grad in zip(expected_grads, result_grads):
                    torch.testing.assert_close(result_grad, expected_grad, rtol=5e-3, atol=5e-3)
                    compared_grads += 1
        assert compared_grads > 0
        monkeypatch.setattr(zero_cls, name, original)
        reference.step()
        monkeypatch.setattr(zero_cls, name, wrapped)
        actual.step()
        for expected, result in zip(reference.module.parameters(), actual.module.parameters()):
            torch.testing.assert_close(result, expected, rtol=5e-3, atol=5e-3)
    if kind == "owner":
        counts = actual.optimizer._turbo_physai_obr["counts"]
        assert counts["reduce_scatter"] > 0, f"no reduce-scatter path observed: {counts}"
    else:
        assert actual.optimizer._turbo_physai_allgather["calls"] == 3


def test_allgather_overflow_and_inflight_exception(monkeypatch):
    from turbo_physai.optimizations.models.groot import zero2_allgather as ag

    if int(os.environ["WORLD_SIZE"]) != 8:
        pytest.skip("AllGather integration gate requires eight ranks")
    rank, _, zero_cls, reference, actual, original, wrapped = _engines(monkeypatch, allgather=True)
    inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    before = [param.detach().clone() for param in actual.module.parameters()]
    for engine in (reference, actual):
        monkeypatch.setattr(engine.optimizer, "check_grad_overflow", True)
        monkeypatch.setattr(engine.optimizer, "check_overflow",
                            lambda engine=engine, **kwargs: setattr(engine.optimizer, "overflow", True))
    monkeypatch.setattr(zero_cls, "step", original)
    reference.backward(reference(inputs).float().square().mean())
    reference.step()
    monkeypatch.setattr(zero_cls, "step", wrapped)
    actual.backward(actual(inputs).float().square().mean())
    actual.step()
    for expected, result in zip(before, actual.module.parameters()):
        torch.testing.assert_close(result, expected)
    for expected, result in zip(reference.module.parameters(), actual.module.parameters()):
        torch.testing.assert_close(result, expected)

    monkeypatch.setattr(actual.optimizer, "check_grad_overflow", False)
    actual.optimizer.overflow = False
    original_launch = ag.launch_group_all_gather
    def fail_after_submit(optimizer, index, comm):
        if index == 1:
            raise RuntimeError("injected failure after first gather")
        return original_launch(optimizer, index, comm)
    monkeypatch.setattr(ag, "launch_group_all_gather", fail_after_submit)
    actual.backward(actual(inputs).float().square().mean())
    with pytest.raises(RuntimeError, match="injected failure"):
        actual.step()
    assert not getattr(actual.optimizer, "_turbo_physai_pending_allgather", None)
    assert actual.optimizer._turbo_physai_allgather_failed
    with pytest.raises(RuntimeError, match="previously failed"):
        actual.optimizer.step()
