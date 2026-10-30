"""Focused behavioral checks for the PointPillars replacements."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_voxel_coordinates_and_truncation():
    voxel = load("voxelize_legacy_sorted")
    points = torch.tensor([[2.1, 1.1, .1, 1], [.1, .1, .1, 2],
                           [2.2, 1.1, .1, 3], [1.1, 1.1, .1, 4]])
    voxels, xyz, counts = voxel.hard_voxelize_legacy_sorted(
        points, [1., 1., 1.], [0., 0., 0., 4., 4., 2.], 2, 2)
    assert xyz.tolist() == [[2, 1, 0], [0, 0, 0]]
    assert counts.tolist() == [2, 1]
    assert torch.equal(voxels[0, :2], points[[0, 2]])


def test_unsupported_voxel_input_uses_original():
    voxel = load("voxelize_legacy_sorted")
    sentinel = object()
    wrapped = voxel.voxelization_layer_forward_wrapper(lambda self, p: sentinel, {})
    layer = SimpleNamespace(max_voxels=(2, 2), training=True, deterministic=True,
                            max_num_points=2, voxel_size=[1., 1., 1.])
    assert wrapped(layer, torch.zeros((1, 4))) is sentinel


def test_compile_disable_switch_skips_compilation(monkeypatch):
    anchor = load("compiled_anchor_grid")
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    monkeypatch.setattr(torch, "compile", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("compile must not be called")))
    original = lambda self, *a, **k: "original"
    assert anchor.compile_anchor_grid_wrapper(original, {})(None) == "original"


def test_compile_config_restored_on_success_and_error():
    wrapper = load("compile_model_modules").compile_pointpillars_modules_wrapper
    runner = SimpleNamespace(cfg={"compile": False})
    def check(runner, target):
        assert runner.cfg["compile"] == {"backend": "inductor"}
        if target == "error":
            raise RuntimeError("test")
        return target
    wrapped = wrapper(check, {})
    assert wrapped(runner, "train_step") == "train_step"
    assert runner.cfg["compile"] is False
    with pytest.raises(RuntimeError):
        wrapped(runner, "error")
    assert runner.cfg["compile"] is False


def test_compile_restores_missing_mmengine_config_key():
    from mmengine.config import Config
    wrapper = load("compile_model_modules").compile_pointpillars_modules_wrapper
    runner = SimpleNamespace(cfg=Config({}))
    assert wrapper(lambda runner, target: target, {})(runner, "train_step") == "train_step"
    assert "compile" not in runner.cfg


def test_cudnn_flags_restored_on_error():
    wrapper = load("cudnn_benchmark").cudnn_benchmark_wrapper
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    def fail():
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
        raise RuntimeError("test")
    with pytest.raises(RuntimeError):
        wrapper(fail, {})()
    assert (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) == (benchmark, deterministic)


def test_dynamo_ddp_scope_is_restored_on_error():
    import torch._dynamo.config as config
    wrapper = load("optimize_ddp").disable_dynamo_ddp_wrapper
    previous = config.optimize_ddp
    def fail():
        assert config.optimize_ddp is False
        raise RuntimeError("test")
    with pytest.raises(RuntimeError):
        wrapper(fail, {})()
    assert config.optimize_ddp == previous
