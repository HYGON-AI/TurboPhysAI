# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.util
import sys
import types

import pytest


torch = pytest.importorskip("torch")

if importlib.util.find_spec("torchvision") is None:
    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    functional = types.ModuleType("torchvision.transforms.functional")
    functional.rotate = lambda tensor, *args, **kwargs: tensor
    transforms.functional = functional
    torchvision.transforms = transforms
    sys.modules.setdefault("torchvision", torchvision)
    sys.modules.setdefault("torchvision.transforms", transforms)
    sys.modules.setdefault("torchvision.transforms.functional", functional)

from turbo_physai.optimizations.models.bevformer import geometry_sca
from turbo_physai.optimizations.models.bevformer import tsa


def test_point_sampling_tensor_projects_identity_camera_coordinates():
    reference_points = torch.tensor(
        [[[[0.5, 0.5, 0.5], [0.25, 0.25, 0.5]]]],
        dtype=torch.float32,
    )
    lidar2img = torch.eye(4).reshape(1, 1, 4, 4)
    projected, mask = geometry_sca._point_sampling_tensor(
        reference_points,
        lidar2img,
        [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        2,
        2,
    )
    expected = torch.tensor(
        [[[[[0.5, 0.5]], [[0.25, 0.25]]]]],
        dtype=torch.float32,
    ).reshape_as(projected)
    torch.testing.assert_close(projected, expected)
    assert mask.all()


def test_get_bev_features_flattens_levels_without_layout_copy():
    captured = {}

    class Encoder:
        def __call__(self, query, key, value, **kwargs):
            captured.update(query=query, key=key, value=value, kwargs=kwargs)
            return query

    model = types.SimpleNamespace(
        use_shift=False,
        rotate_prev_bev=False,
        rotate_center=None,
        use_can_bus=False,
        can_bus_mlp=lambda tensor: torch.zeros(tensor.shape[0], 2),
        use_cams_embeds=True,
        cams_embeds=torch.zeros(2, 2),
        level_embeds=torch.zeros(1, 2),
        encoder=Encoder(),
    )
    features = [torch.arange(8.0).reshape(1, 2, 2, 1, 2)]
    queries = torch.zeros(4, 2)
    position = torch.zeros(1, 2, 2, 2)
    metadata = [{"can_bus": [0.0, 0.0, 0.0, 0.0]}]
    actual = geometry_sca.get_bev_features(
        model,
        features,
        queries,
        2,
        2,
        bev_pos=position,
        img_metas=metadata,
    )
    assert actual.shape == (4, 1, 2)
    assert captured["key"].shape == (1, 2, 2, 2)
    assert captured["value"] is captured["key"]
    assert captured["kwargs"]["spatial_shapes"].tolist() == [[1, 2]]
    assert captured["kwargs"]["level_start_index"].tolist() == [0]
    torch.testing.assert_close(
        captured["kwargs"]["shift"], torch.zeros(1, 2)
    )


def test_spatial_cross_attention_scatter_average_and_backward():
    class Attention:
        def __call__(self, *, query, **kwargs):
            del kwargs
            return query

    model = types.SimpleNamespace(
        num_cams=2,
        embed_dims=2,
        deformable_attention=Attention(),
        output_proj=torch.nn.Identity(),
        dropout=torch.nn.Identity(),
    )
    query = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
        requires_grad=True,
    )
    value = torch.zeros(1, 2, 1, 2)
    indexes = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
    lengths = torch.tensor([[3, 3]])
    reference = torch.zeros(2, 1, 3, 1, 2)
    mask = torch.ones(2, 1, 3, 1, dtype=torch.bool)
    actual = geometry_sca.spatial_cross_attention_forward(
        model,
        query,
        key=value,
        value=value,
        reference_points_cam=reference,
        bev_mask=mask,
        indexes=indexes,
        index_lengths=lengths,
        spatial_shapes=torch.tensor([[1, 1]]),
        level_start_index=torch.tensor([0]),
    )
    torch.testing.assert_close(actual, query * 2)
    actual.sum().backward()
    torch.testing.assert_close(query.grad, torch.full_like(query, 2))


def test_temporal_self_attention_cpu_path_preserves_reference_contract(
    monkeypatch,
):
    mmcv = types.ModuleType("mmcv")
    ops = types.ModuleType("mmcv.ops")
    msda = types.ModuleType("mmcv.ops.multi_scale_deform_attn")

    def reference(value, spatial_shapes, sampling_locations, weights):
        del spatial_shapes, sampling_locations, weights
        return value.reshape(value.shape[0], value.shape[1], -1)

    msda.multi_scale_deformable_attn_pytorch = reference
    ops.multi_scale_deform_attn = msda
    mmcv.ops = ops
    monkeypatch.setitem(sys.modules, "mmcv", mmcv)
    monkeypatch.setitem(sys.modules, "mmcv.ops", ops)
    monkeypatch.setitem(sys.modules, msda.__name__, msda)

    model = types.SimpleNamespace(
        batch_first=True,
        embed_dims=2,
        num_bev_queue=2,
        num_heads=1,
        num_levels=1,
        num_points=1,
        value_proj=torch.nn.Identity(),
        sampling_offsets=torch.nn.Linear(4, 4),
        attention_weights=torch.nn.Linear(4, 2),
        output_proj=torch.nn.Identity(),
        dropout=torch.nn.Identity(),
        im2col_step=1,
    )
    torch.nn.init.zeros_(model.sampling_offsets.weight)
    torch.nn.init.zeros_(model.sampling_offsets.bias)
    torch.nn.init.zeros_(model.attention_weights.weight)
    torch.nn.init.zeros_(model.attention_weights.bias)
    query = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    reference_points = torch.full((2, 2, 1, 2), 0.5)
    actual = tsa.temporal_self_attention_forward(
        model,
        query,
        reference_points=reference_points,
        spatial_shapes=torch.tensor([[1, 2]]),
        level_start_index=torch.tensor([0]),
    )
    torch.testing.assert_close(actual, query * 2)
    actual.sum().backward()
    assert torch.count_nonzero(query.grad) == query.numel()
