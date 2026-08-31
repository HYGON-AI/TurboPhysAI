# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from turbo_physai.optimizations.models.bevformer import backbone
from turbo_physai.optimizations.models.bevformer import compile as compile_optimization
from turbo_physai.optimizations.models.bevformer import grid_mask
from turbo_physai.optimizations.models.bevformer import training


def test_extract_img_feat_preserves_batch_camera_layout_and_gradients():
    calls = {}

    class Model:
        use_grid_mask = False
        with_img_neck = False

        @staticmethod
        def img_backbone(image):
            calls["channels_last"] = image.is_contiguous(
                memory_format=torch.channels_last
            )
            return [image * 2]

    image = torch.arange(24.0).reshape(2, 3, 1, 2, 2).requires_grad_(True)
    result = backbone.extract_img_feat(Model(), image, img_metas=[{}, {}])
    assert calls["channels_last"] is True
    assert len(result) == 1
    assert result[0].shape == image.shape
    torch.testing.assert_close(result[0], image * 2)
    result[0].sum().backward()
    torch.testing.assert_close(image.grad, torch.full_like(image, 2))


def test_compiled_extract_img_feat_honors_global_disable(monkeypatch):
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert backbone.compiled_extract_img_feat(object(), {}) is backbone.extract_img_feat

    calls = []
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0")
    monkeypatch.setattr(
        backbone.torch,
        "compile",
        lambda function, **kwargs: calls.append((function, kwargs)) or function,
    )
    compiled = backbone.compiled_extract_img_feat(object(), {})
    assert compiled is backbone.extract_img_feat
    assert calls == [
        (backbone.extract_img_feat, {"mode": "max-autotune-no-cudagraphs"})
    ]


def test_grid_mask_fast_path_is_deterministic_with_fixed_random_values(
    monkeypatch,
):
    values = iter((2, 0, 0))
    monkeypatch.setattr(grid_mask.np.random, "rand", lambda: 0.0)
    monkeypatch.setattr(
        grid_mask.np.random,
        "randint",
        lambda *args, **kwargs: next(values),
    )
    model = types.SimpleNamespace(
        prob=1.0,
        training=True,
        ratio=0.5,
        rotate=1,
        use_h=True,
        use_w=True,
        mode=0,
        offset=False,
    )
    image = torch.ones(1, 1, 4, 4)
    actual = grid_mask.grid_mask_forward(model, image)
    expected_vector = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected = (
        expected_vector[:, None] * expected_vector[None, :]
    ).reshape(1, 1, 4, 4)
    torch.testing.assert_close(actual, expected)

    model.training = False
    assert grid_mask.grid_mask_forward(model, image) is image


def test_compile_wrapper_calls_compiled_function_and_can_be_disabled(monkeypatch):
    def original(value, scale=1):
        return value * scale

    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert compile_optimization.compile_wrapper(original, {}) is original

    calls = []
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0")
    monkeypatch.setattr(
        torch,
        "compile",
        lambda function, **kwargs: (
            calls.append((function, kwargs))
            or (lambda *args, **kw: function(*args, **kw) + 1)
        ),
    )
    wrapped = compile_optimization.compile_wrapper(original, {})
    assert wrapped(3, scale=2) == 7
    assert calls[0][0] is original
    assert calls[0][1]["mode"] == "max-autotune-no-cudagraphs"


def test_training_runtime_wrapper_applies_runtime_policy(monkeypatch):
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    class Model:
        def __init__(self):
            self.memory_format = None

        def to(self, *, memory_format):
            self.memory_format = memory_format
            return self

    cfg = AttrDict(
        optimizer=AttrDict(type="AdamW"),
        data=AttrDict(workers_per_gpu=3),
        custom_hooks=[],
    )
    model = Model()
    calls = []

    def original(actual_model, dataset, actual_cfg, marker=None):
        calls.append((actual_model, dataset, actual_cfg, marker))
        return "trained"

    monkeypatch.setenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "")
    monkeypatch.setenv("TURBO_PHYSAI_WORKERS_PER_GPU", "5")
    monkeypatch.setenv("ENABLE_TORCH_PROFILER", "0")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    wrapped = training.training_runtime_wrapper(original, {})
    assert wrapped(model, "dataset", cfg, marker="value") == "trained"
    assert cfg.optimizer["fused"] is True
    assert cfg.data.samples_per_gpu == 2
    assert cfg.data.workers_per_gpu == 5
    assert model.memory_format is torch.channels_last
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False
    assert calls == [(model, "dataset", cfg, "value")]


def test_training_runtime_wrapper_rejects_non_adamw(monkeypatch):
    class AttrDict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    cfg = AttrDict(
        optimizer=AttrDict(type="SGD"),
        data=AttrDict(workers_per_gpu=1),
    )
    monkeypatch.setenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "")
    wrapped = training.training_runtime_wrapper(lambda *args: None, {})
    with pytest.raises(ValueError, match="optimizer.type=AdamW"):
        wrapped(types.SimpleNamespace(), None, cfg)
