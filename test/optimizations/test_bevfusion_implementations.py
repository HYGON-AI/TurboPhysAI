# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
import types

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.optimizations.models.bevfusion import backbone
from turbo_physai.optimizations.models.bevfusion import compile as compile_optimization
from turbo_physai.optimizations.models.bevfusion import depth
from turbo_physai.optimizations.models.bevfusion import training
from turbo_physai.optimizations.models.bevfusion import transfusion


def test_extract_camera_features_preserves_camera_contract(monkeypatch):
    captured = {}

    class CameraBackbone(torch.nn.Module):
        def forward(self, image):
            captured["channels_last"] = image.is_contiguous(
                memory_format=torch.channels_last
            )
            return image + 1

    class CameraNeck(torch.nn.Module):
        def forward(self, image):
            return [image * 2]

    class ViewTransform(torch.nn.Module):
        def forward(self, image, *args, depth_loss, gt_depths):
            captured["view_shape"] = image.shape
            captured["depth_loss"] = depth_loss
            captured["gt_depths"] = gt_depths
            return image.sum(dim=1)

    model = types.SimpleNamespace(
        encoders={
            "camera": {
                "backbone": CameraBackbone(),
                "neck": CameraNeck(),
                "vtransform": ViewTransform(),
            }
        },
        use_depth_loss=True,
    )
    monkeypatch.setenv("MMDET3D_CHANNELS_LAST", "1")
    image = torch.arange(16.0).reshape(1, 2, 2, 2, 2).requires_grad_(True)
    arguments = [None] * 11
    actual = backbone.extract_camera_features(
        model,
        image,
        *arguments,
        gt_depths="depth-target",
    )
    assert captured["channels_last"] is True
    assert captured["view_shape"] == (1, 2, 2, 2, 2)
    assert captured["depth_loss"] is True
    assert captured["gt_depths"] == "depth-target"
    assert actual.shape == (1, 2, 2, 2)
    actual.sum().backward()
    assert torch.count_nonzero(image.grad) == image.numel()


def test_parse_losses_reduces_values_and_preserves_autograd():
    first = torch.tensor([1.0, 3.0], requires_grad=True)
    second = torch.tensor([2.0, 4.0], requires_grad=True)
    metric = torch.tensor([7.0, 9.0])
    loss, log_vars = training.parse_losses(
        None,
        {
            "loss_first": first,
            "loss_second": [second],
            "metric": metric,
        },
    )
    torch.testing.assert_close(loss, torch.tensor(5.0))
    assert log_vars == {
        "loss_first": 2.0,
        "loss_second": 3.0,
        "metric": 8.0,
        "loss": 5.0,
    }
    loss.backward()
    torch.testing.assert_close(first.grad, torch.full_like(first, 0.5))
    torch.testing.assert_close(second.grad, torch.full_like(second, 0.5))

    with pytest.raises(TypeError, match="not a tensor"):
        training.parse_losses(None, {"loss_invalid": object()})


def test_depth_lss_get_cam_feats_returns_equivalent_factors_and_gradients():
    class Transform:
        D = 2
        C = 1

        @staticmethod
        def dtransform(value):
            return torch.cat((value, value), dim=1)

        @staticmethod
        def depthnet(value):
            return value

    image = torch.tensor([[[[[3.0, 4.0]]]]], requires_grad=True)
    depth_input = torch.tensor([[[[[1.0, 2.0]]]]], requires_grad=True)
    factors = depth.depth_lss_get_cam_feats(
        Transform(), image, depth_input
    )
    expected_depth = torch.full((1, 2, 1, 2), 0.5)
    expected_features = torch.tensor([[[[3.0], [4.0]]]])
    torch.testing.assert_close(factors.depth, expected_depth)
    torch.testing.assert_close(factors.features, expected_features)
    (factors.depth.sum() + factors.features.sum()).backward()
    torch.testing.assert_close(image.grad, torch.ones_like(image))
    assert depth_input.grad is not None


def test_transfusion_forward_single_builds_queries_and_keeps_dense_heatmap_grad():
    class HeatmapHead(torch.nn.Module):
        def forward(self, value):
            return value[:, :1]

    class Decoder(torch.nn.Module):
        def forward(self, query, lidar, query_pos, bev_pos):
            del lidar, query_pos, bev_pos
            return query

    class PredictionHead(torch.nn.Module):
        def forward(self, query):
            batch, _, proposals = query.shape
            return {
                "center": query.new_zeros(batch, 2, proposals),
                "heatmap": query.new_zeros(batch, 1, proposals),
            }

    class ZeroClassEncoding(torch.nn.Module):
        def forward(self, one_hot):
            return one_hot.new_zeros(one_hot.shape[0], 2, one_hot.shape[-1])

    model = types.SimpleNamespace(
        shared_conv=torch.nn.Identity(),
        bev_pos=torch.tensor(
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
              [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
              [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]]]
        ),
        heatmap_head=HeatmapHead(),
        nms_kernel_size=3,
        test_cfg={"dataset": "Other"},
        num_proposals=2,
        num_classes=1,
        class_encoding=ZeroClassEncoding(),
        num_decoder_layers=1,
        decoder=[Decoder()],
        prediction_heads=[PredictionHead()],
        auxiliary=False,
    )
    inputs = torch.tensor(
        [[[[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 3.0]],
          [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        requires_grad=True,
    )
    result = transfusion.transfusion_forward_single(
        model, inputs, None, None
    )[0]
    assert result["center"].shape == (1, 2, 2)
    assert result["query_heatmap_score"].shape == (1, 1, 2)
    assert result["dense_heatmap"].shape == (1, 1, 3, 3)
    assert model.query_labels.tolist() == [[0, 0]]
    result["dense_heatmap"].sum().backward()
    assert torch.count_nonzero(inputs.grad[:, :1]) == 9


def test_transfusion_get_targets_aggregates_samples(monkeypatch):
    mmdet = types.ModuleType("mmdet")
    core = types.ModuleType("mmdet.core")

    def multi_apply(function, *iterables):
        rows = [function(*items) for items in zip(*iterables)]
        return tuple([row[index] for row in rows] for index in range(len(rows[0])))

    core.multi_apply = multi_apply
    mmdet.core = core
    monkeypatch.setitem(sys.modules, "mmdet", mmdet)
    monkeypatch.setitem(sys.modules, "mmdet.core", core)

    class Box:
        def __init__(self, value):
            self.tensor = torch.tensor([[value, 0.0]])

    class Model:
        @staticmethod
        def get_targets_single(box, label, prediction, batch_index, tensor):
            del box, prediction, tensor
            value = int(batch_index) + 1
            scalar = torch.tensor(float(value))
            return (
                torch.tensor([[int(label[0])]]),
                torch.ones(1, 1),
                torch.full((1, 1, 2), float(value)),
                torch.ones(1, 1, 2),
                torch.full((1, 1), 0.5),
                value,
                scalar,
                torch.full((1, 1, 1, 1), float(value)),
            )

    boxes = [Box(1.0), Box(2.0)]
    labels = [torch.tensor([0]), torch.tensor([1])]
    predictions = [{"center": torch.zeros(2, 2, 1)}]
    result = transfusion.transfusion_get_targets(
        Model(), boxes, labels, predictions
    )
    assert result[0].shape == (2, 1)
    assert result[2].shape == (2, 1, 2)
    assert result[5] == 3
    torch.testing.assert_close(result[6], torch.tensor(1.5))
    assert result[7].shape == (2, 1, 1, 1)


def test_transfusion_loss_builds_all_loss_terms_and_backpropagates():
    class Model:
        num_decoder_layers = 1
        auxiliary = False
        num_proposals = 2
        num_classes = 1
        train_cfg = {"code_weights": [1.0] * 8}

        @staticmethod
        def get_targets(*args):
            del args
            return (
                torch.tensor([[0, 1]]),
                torch.ones(1, 2),
                torch.zeros(1, 2, 8),
                torch.ones(1, 2, 8),
                torch.ones(1, 2),
                1,
                torch.tensor(0.75),
                torch.zeros(1, 1, 1, 2),
            )

        @staticmethod
        def loss_heatmap(prediction, target, avg_factor):
            return (prediction - target).square().sum() / avg_factor

        @staticmethod
        def loss_cls(score, labels, weights, avg_factor):
            del labels
            return (score.square().sum() * weights.mean()) / avg_factor

        @staticmethod
        def loss_bbox(encoded, targets, weights, avg_factor):
            return ((encoded - targets).square() * weights).sum() / avg_factor

    dense_heatmap = torch.zeros(1, 1, 1, 2, requires_grad=True)
    predictions = {
        # _clip_sigmoid mutates its input, matching the non-leaf model output.
        "dense_heatmap": dense_heatmap + 0,
        "heatmap": torch.ones(1, 1, 2, requires_grad=True),
        "center": torch.ones(1, 2, 2, requires_grad=True),
        "height": torch.ones(1, 1, 2, requires_grad=True),
        "dim": torch.ones(1, 3, 2, requires_grad=True),
        "rot": torch.ones(1, 2, 2, requires_grad=True),
    }
    losses = transfusion.transfusion_loss(
        Model(), [], [], [[predictions]]
    )
    assert set(losses) == {
        "loss_heatmap",
        "layer_-1_loss_cls",
        "layer_-1_loss_bbox",
        "matched_ious",
    }
    total = sum(value for name, value in losses.items() if "loss" in name)
    total.backward()
    assert dense_heatmap.grad is not None
    assert predictions["center"].grad is not None


def test_compile_wrappers_preserve_callable_and_class_contract(monkeypatch):
    calls = []

    def fake_compile(target, **kwargs):
        calls.append((target, kwargs))
        return target

    monkeypatch.setattr(torch, "compile", fake_compile)

    def function(value):
        return value + 1

    class Module(torch.nn.Module):
        def forward(self, value):
            return value + 2

    wrapped = compile_optimization.compile_wrapper(function, {"mode": "mode-a"})
    compiled_class = compile_optimization.compile_class_wrapper(
        Module, {"mode": "mode-b"}
    )
    assert wrapped(2) == 3
    assert compiled_class is Module
    assert calls == [
        (function, {"mode": "mode-a"}),
        (Module, {"mode": "mode-b"}),
    ]


def test_training_compile_honors_global_disable(monkeypatch):
    class Model:
        def __init__(self):
            self.backbone = object()
            self.encoders = {"camera": {"backbone": self.backbone}}

    model = Model()
    calls = []

    def fake_compile(target, **kwargs):
        calls.append((target, kwargs))
        return "compiled-backbone"

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert training._compile_model(model, "test-mode") is model
    assert model.encoders["camera"]["backbone"] is model.backbone
    assert calls == []

    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0")
    assert training._compile_model(model, "test-mode") is model
    assert model.encoders["camera"]["backbone"] == "compiled-backbone"
    assert calls == [
        (
            model.backbone,
            {"mode": "test-mode", "fullgraph": False, "dynamic": False},
        )
    ]


def test_training_wrapper_applies_channels_last_and_compile_policy(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.camera_backbone = torch.nn.Conv2d(1, 1, 1)
            self.encoders = {"camera": {"backbone": self.camera_backbone}}
            self.decoder = {}

    model = Model()
    cfg = types.SimpleNamespace(logger=None)
    captured = []

    def original(actual_model, dataset, actual_cfg):
        captured.append((actual_model, dataset, actual_cfg))
        return "trained"

    monkeypatch.setattr(torch.multiprocessing, "get_start_method", lambda **kwargs: None)
    monkeypatch.setattr(torch.multiprocessing, "set_start_method", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        training,
        "_compile_model",
        lambda actual_model, mode: captured.append(mode) or actual_model,
    )
    wrapped = training.training_wrapper(
        original,
        {
            "channels_last": True,
            "compile_mode": "test-mode",
            "cuda_prefetch": False,
            "start_method": "fork",
        },
    )
    assert wrapped(model, "dataset", cfg) == "trained"
    assert model.encoders["camera"]["backbone"].weight.is_contiguous(
        memory_format=torch.channels_last
    )
    assert captured[0] == "test-mode"
    assert captured[1] == (model, "dataset", cfg)
