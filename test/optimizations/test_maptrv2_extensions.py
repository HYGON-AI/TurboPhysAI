# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import sys
import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")


def _reference_linear_sum_assignment(cost):
    """Exhaustive optimum for the small matrices used by these tests."""

    values = cost.tolist()
    num_rows = len(values)
    num_columns = len(values[0]) if num_rows else 0
    if not num_rows or not num_columns:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )

    best = None
    if num_rows <= num_columns:
        for rows in itertools.permutations(range(num_rows)):
            for columns in itertools.permutations(range(num_columns), num_rows):
                score = sum(
                    values[row][column]
                    for row, column in zip(rows, columns)
                )
                if best is None or score < best[0]:
                    best = (score, list(rows), list(columns))
    else:
        for columns in itertools.permutations(range(num_columns)):
            for rows in itertools.permutations(range(num_rows), num_columns):
                score = sum(
                    values[row][column]
                    for row, column in zip(rows, columns)
                )
                if best is None or score < best[0]:
                    best = (score, list(rows), list(columns))
    return (
        np.asarray(best[1], dtype=np.int64),
        np.asarray(best[2], dtype=np.int64),
    )


def _install_scipy_assignment_stub(monkeypatch):
    scipy = types.ModuleType("scipy")
    scipy.__path__ = []
    optimize = types.ModuleType("scipy.optimize")
    optimize.linear_sum_assignment = _reference_linear_sum_assignment
    scipy.optimize = optimize
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)


def _install_padded_contract_stubs(monkeypatch):
    project_names = (
        "projects",
        "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.maptr",
        "projects.mmdet3d_plugin.maptr.dense_heads",
        "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head",
    )
    project_modules = {
        name: types.ModuleType(name) for name in project_names
    }
    for parent, child in zip(project_names, project_names[1:]):
        setattr(
            project_modules[parent],
            child.rsplit(".", 1)[1],
            project_modules[child],
        )
    head_module = project_modules[project_names[-1]]
    head_module.normalize_2d_bbox = lambda value, _: value
    head_module.normalize_2d_pts = lambda value, _: value
    head_module.normalize_3d_pts = lambda value, _: value
    head_module.denormalize_2d_bbox = lambda value, _: value

    mmdet_names = (
        "mmdet",
        "mmdet.core",
        "mmdet.core.bbox",
        "mmdet.core.bbox.assigners",
    )
    mmdet_modules = {
        name: types.ModuleType(name) for name in mmdet_names
    }
    for parent, child in zip(mmdet_names, mmdet_names[1:]):
        setattr(
            mmdet_modules[parent],
            child.rsplit(".", 1)[1],
            mmdet_modules[child],
        )

    class AssignResult:
        def __init__(
            self, num_gts, assigned_gt_inds, max_overlaps=None, labels=None
        ):
            self.num_gts = num_gts
            self.assigned_gt_inds = assigned_gt_inds
            self.max_overlaps = max_overlaps
            self.labels = labels

    def multi_apply(function, *args):
        return tuple(
            map(list, zip(*(function(*values) for values in zip(*args))))
        )

    mmdet_modules["mmdet.core"].multi_apply = multi_apply
    mmdet_modules["mmdet.core.bbox.assigners"].AssignResult = AssignResult

    for name, module in project_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name, module in mmdet_modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _install_lidar_instance_lines_stub(monkeypatch):
    module_names = (
        "projects",
        "projects.mmdet3d_plugin",
        "projects.mmdet3d_plugin.datasets",
        "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset",
    )
    modules = {name: types.ModuleType(name) for name in module_names}
    for parent, child in zip(module_names, module_names[1:]):
        setattr(modules[parent], child.rsplit(".", 1)[1], modules[child])

    class LiDARInstanceLines:
        def __init__(
            self,
            instance_line_list,
            instance_labels,
            sample_dist=1,
            num_samples=250,
            padding=False,
            fixed_num=-1,
            padding_value=-10000,
            patch_size=None,
        ):
            self.received = {
                "instance_line_list": instance_line_list,
                "instance_labels": instance_labels,
                "sample_dist": sample_dist,
                "num_samples": num_samples,
                "padding": padding,
                "fixed_num": fixed_num,
                "padding_value": padding_value,
                "patch_size": patch_size,
            }
            self.instances = instance_line_list
            self.labels = instance_labels

    modules[module_names[-1]].LiDARInstanceLines = LiDARInstanceLines
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return LiDARInstanceLines


def test_pad_to_static_list_preserves_values_and_mask():
    from turbo_physai.optimizations.models.maptrv2 import assigner

    first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padded, mask, length = assigner.pad_to_static_list([first])[0]
    assert padded.shape == (assigner.STATIC_GT_COUNT, 2)
    assert length == 2
    torch.testing.assert_close(padded[:2], first)
    assert mask[:2].all()
    assert not mask[2:].any()


def test_static_target_path_uses_padded_gt_contract():
    from turbo_physai.optimizations.models.maptrv2 import assigner

    class Assigner:
        @staticmethod
        def assign(*args):
            del args
            return object(), torch.tensor([[0]])

    class Sampler:
        @staticmethod
        def sample(assign_result, bbox_pred, gt_bboxes):
            del assign_result, bbox_pred, gt_bboxes
            return types.SimpleNamespace(
                pos_inds=torch.tensor([0]),
                neg_inds=torch.tensor([1]),
                pos_assigned_gt_inds=torch.tensor([0]),
                pos_gt_bboxes=torch.tensor([[2.0, 3.0, 4.0, 5.0]]),
            )

    model = types.SimpleNamespace(
        assigner=Assigner(),
        sampler=Sampler(),
        num_classes=3,
    )
    bbox_pred = torch.zeros(2, 4)
    result = assigner.get_target_single(
        model,
        cls_score=torch.zeros(2, 3),
        bbox_pred=bbox_pred,
        pts_pred=torch.zeros(2, 2, 2),
        gt_labels=(torch.tensor([1]), torch.tensor([True]), 1),
        gt_bboxes=(
            torch.tensor([[1.0, 1.0, 2.0, 2.0]]),
            torch.tensor([True]),
            1,
        ),
        gt_shifts_pts=(
            torch.tensor([[[[7.0, 8.0], [9.0, 10.0]]]]),
            torch.tensor([True]),
            1,
        ),
    )
    labels, _, _, _, pts_targets, _, _, _ = result
    assert labels.tolist() == [1, 3]
    torch.testing.assert_close(
        pts_targets[0], torch.tensor([[7.0, 8.0], [9.0, 10.0]])
    )


@pytest.mark.parametrize("num_gts", (0, 1, 3, 6))
def test_assign_impl_matches_unpadded_assignment(monkeypatch, num_gts):
    from turbo_physai.optimizations.models.maptrv2 import assigner

    _install_scipy_assignment_stub(monkeypatch)
    _install_padded_contract_stubs(monkeypatch)
    monkeypatch.setattr(
        assigner, "_get_hungarian_match",
        lambda: assigner._hungarian_match_impl,
    )

    num_queries = 4
    num_points = 3
    num_orders = 2
    static_count = assigner.STATIC_GT_COUNT
    generator = torch.Generator().manual_seed(17)
    padded_bboxes = torch.full((static_count, 4), -1000.0)
    padded_labels = torch.full((static_count,), -7, dtype=torch.long)
    padded_points = torch.full(
        (static_count, num_orders, num_points, 2), -1000.0
    )
    valid_mask = torch.zeros(static_count, dtype=torch.bool)
    if num_gts:
        padded_bboxes[:num_gts] = torch.rand(
            num_gts, 4, generator=generator
        )
        padded_labels[:num_gts] = torch.randint(
            0, 3, (num_gts,), generator=generator
        )
        padded_points[:num_gts] = torch.rand(
            num_gts, num_orders, num_points, 2,
            generator=generator,
        )
        valid_mask[:num_gts] = True

    cls_cost = torch.full((num_queries, static_count), -1000.0)
    pts_cost = torch.full(
        (num_queries, static_count, num_orders), -1000.0
    )
    reg_cost = torch.rand(num_queries, 1, generator=generator)
    iou_cost = torch.rand(num_queries, 1, generator=generator)
    if num_gts:
        cls_cost[:, :num_gts] = torch.rand(
            num_queries, num_gts, generator=generator
        )
        pts_cost[:, :num_gts, :] = torch.rand(
            num_queries, num_gts, num_orders,
            generator=generator,
        )

    expected_order = pts_cost.argmin(dim=2)
    total_cost = (
        cls_cost + reg_cost + iou_cost + pts_cost.min(dim=2).values
    )
    expected_rows, expected_columns = (
        _reference_linear_sum_assignment(
            total_cost[:, valid_mask].numpy()
        )
    )
    expected_inds = torch.zeros(num_queries, dtype=torch.long)
    expected_labels = torch.full((num_queries,), -1, dtype=torch.long)
    if expected_rows.size:
        expected_rows_tensor = torch.as_tensor(expected_rows)
        expected_columns_tensor = torch.as_tensor(expected_columns)
        expected_inds[expected_rows_tensor] = expected_columns_tensor + 1
        expected_labels[expected_rows_tensor] = padded_labels[
            expected_columns_tensor
        ]

    class Cost:
        weight = 1.0

        def __call__(self, left, right):
            del left, right
            return cls_cost

    class ScalarCost:
        weight = 1.0

        def __init__(self, value):
            self.value = value

        def __call__(self, left, right):
            del left, right
            return self.value

    class PointsCost:
        weight = 1.0

        def __call__(self, left, right):
            del left, right
            return pts_cost

    model = types.SimpleNamespace(
        cls_cost=Cost(),
        reg_cost=ScalarCost(reg_cost),
        iou_cost=ScalarCost(iou_cost),
        pts_cost=PointsCost(),
        z_cfg={"gt_z_flag": False},
        pc_range=[0.0, 0.0, -1.0, 100.0, 100.0, 1.0],
    )

    assign_result, order_index = assigner._assign_impl(
        model,
        torch.zeros(num_queries, 4),
        torch.zeros(num_queries, 3),
        torch.zeros(num_queries, num_points, 2),
        (padded_bboxes, valid_mask, num_gts),
        (padded_labels, valid_mask, num_gts),
        (padded_points, valid_mask, num_gts),
    )

    assert assign_result.num_gts == num_gts
    torch.testing.assert_close(
        assign_result.assigned_gt_inds, expected_inds
    )
    torch.testing.assert_close(assign_result.labels, expected_labels)
    torch.testing.assert_close(
        order_index[:, :num_gts], expected_order[:, :num_gts]
    )


def test_static_loss_uses_padded_contract(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2 import assigner

    _install_padded_contract_stubs(monkeypatch)
    calls = []

    class Model:
        gt_shift_pts_pattern = "v0"
        aux_seg = {"use_aux_seg": False}

        @staticmethod
        def loss_single(
            cls_scores,
            bbox_preds,
            pts_preds,
            gt_bboxes,
            gt_labels,
            gt_points,
            gt_bboxes_ignore,
        ):
            calls.append(
                {
                    "cls_scores": cls_scores,
                    "bbox_preds": bbox_preds,
                    "pts_preds": pts_preds,
                    "gt_bboxes": gt_bboxes,
                    "gt_labels": gt_labels,
                    "gt_points": gt_points,
                    "gt_bboxes_ignore": gt_bboxes_ignore,
                }
            )
            base = len(calls) * 10
            return tuple(
                torch.tensor(float(base + index)) for index in range(5)
            )

    gt_bboxes = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )
    gt_points = torch.arange(
        2 * 2 * 3 * 2, dtype=torch.float32
    ).reshape(2, 2, 3, 2)
    gt_vecs = [
        types.SimpleNamespace(
            bbox=gt_bboxes,
            shift_fixed_num_sampled_points=gt_points,
        )
    ]
    gt_labels = [torch.tensor([0, 1])]
    preds_dicts = {
        "all_cls_scores": [torch.zeros(1, 4, 3), torch.ones(1, 4, 3)],
        "all_bbox_preds": [torch.zeros(1, 4, 4), torch.ones(1, 4, 4)],
        "all_pts_preds": [
            torch.zeros(1, 4, 3, 2),
            torch.ones(1, 4, 3, 2),
        ],
        "enc_cls_scores": torch.zeros(1, 4, 3),
        "enc_bbox_preds": torch.zeros(1, 4, 4),
        "enc_pts_preds": torch.zeros(1, 4, 3, 2),
        "seg": None,
        "pv_seg": None,
    }

    loss_dict = assigner._static_loss_impl(
        Model(),
        gt_vecs,
        gt_labels,
        None,
        None,
        preds_dicts,
    )

    assert len(calls) == 3
    for call in calls:
        padded_bboxes, bbox_mask, bbox_length = call["gt_bboxes"][0]
        padded_labels, label_mask, label_length = call["gt_labels"][0]
        padded_points, point_mask, point_length = call["gt_points"][0]
        assert padded_bboxes.shape == (assigner.STATIC_GT_COUNT, 4)
        assert padded_labels.shape == (assigner.STATIC_GT_COUNT,)
        assert padded_points.shape == (
            assigner.STATIC_GT_COUNT, 2, 3, 2
        )
        assert torch.equal(bbox_mask, label_mask)
        assert torch.equal(bbox_mask, point_mask)
        assert (bbox_length, label_length, point_length) == (2, 2, 2)
        assert bbox_mask[:2].all()
        assert not bbox_mask[2:].any()
        assert call["gt_bboxes_ignore"] is None

    torch.testing.assert_close(calls[0]["gt_bboxes"][0][0][:2], gt_bboxes)
    torch.testing.assert_close(calls[0]["gt_points"][0][0][:2], gt_points)
    assert torch.count_nonzero(calls[2]["gt_labels"][0][0]) == 0
    assert loss_dict["loss_cls"].item() == 20.0
    assert loss_dict["enc_loss_cls"].item() == 30.0


def test_assigner_is_eager_by_default(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2 import assigner

    captured = []
    monkeypatch.delenv("TURBO_PHYSAI_MAPTRV2_ASSIGN_COMPILE", raising=False)
    monkeypatch.setattr(
        assigner,
        "_assign_no_grad",
        lambda *args: captured.append(args) or "assignment",
    )
    assert assigner.assign(None, 1, 2, 3, 4, 5, 6, 7, 8) == "assignment"
    assert captured


def test_reference_module_exposes_all_eight_compile_helpers():
    from turbo_physai.optimizations.models.maptrv2 import reference

    assert set(reference._HELPERS) == {
        "matmul_1",
        "matmul_2",
        "matmul_3",
        "extract_metas",
        "down_sample",
        "initialize_queries_and_bev",
        "compute_decoder_predictions",
        "prepare_transformer_inputs",
    }


def test_reference_wrappers_compile_each_helper_once(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2 import reference

    calls = []

    def fake_compile(function, **kwargs):
        calls.append((function, kwargs))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        reference, "_force_fp32", lambda function, apply_to=None: function
    )
    monkeypatch.setattr(reference, "_COMPILED", {})
    monkeypatch.setattr(reference, "_ACTIVE_MODE", None)
    reference.base_transform_get_geometry_wrapper(
        object(), {"mode": "test-mode"}
    )
    reference.base_transform_forward_wrapper(
        object(), {"mode": "test-mode"}
    )
    assert len(calls) == 8
    assert {kwargs["mode"] for _, kwargs in calls} == {"test-mode"}


def test_reference_matmul_1_keeps_the_coordinate_axis():
    from turbo_physai.optimizations.models.maptrv2 import reference

    batch, cameras = 1, 2
    rotation = torch.eye(3).view(1, 1, 3, 3).expand(
        batch, cameras, -1, -1
    )
    points = torch.tensor(
        [2.0, 3.0, 4.0], dtype=torch.float32
    ).reshape(1, 1, 1, 1, 1, 3).expand(
        batch, cameras, 1, 1, 1, 3
    )
    result = reference.matmul_1(
        None, rotation, points, torch.zeros(batch, cameras, 3)
    )
    assert result.shape == (batch, cameras, 1, 1, 1, 3, 1)
    torch.testing.assert_close(
        result.reshape(batch, cameras, 3),
        torch.tensor([[[8.0, 12.0, 4.0], [8.0, 12.0, 4.0]]]),
    )


def test_reference_down_sample_collapses_depth_to_channels():
    from turbo_physai.optimizations.models.maptrv2 import reference

    model = types.SimpleNamespace(downsample=torch.nn.Identity())
    value = torch.zeros(2, 3, 4, 5, 6)
    result = reference.down_sample(model, value)
    assert result.shape == (2, 18, 5, 4)


def test_reference_bev_pool_accepts_rocm_hw_layout(monkeypatch):
    from turbo_physai.optimizations.models.maptrv2 import reference

    module_names = ("mmdet3d", "mmdet3d.ops")
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["mmdet3d"].ops = modules["mmdet3d.ops"]
    expected = torch.zeros(4, 1, 200, 400, 256)
    modules["mmdet3d.ops"].bev_pool = lambda *args: expected
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    model = types.SimpleNamespace(
        C=256,
        nx=torch.tensor([200, 400, 1]),
        bx=torch.tensor([0.0, 0.0, 0.0]),
        dx=torch.tensor([1.0, 1.0, 1.0]),
    )
    features = torch.zeros(4, 1, 1, 2, 2, 256)
    geometry = torch.ones(4, 1, 1, 2, 2, 3)
    result = reference._bev_pool_5d(model, features, geometry)
    assert result is expected


def test_reference_bev_pool_uses_installed_fix():
    from turbo_physai.optimizations.models.maptrv2 import reference

    expected = torch.zeros(1, 1, 2, 3, 4)
    calls = []

    class Model:
        def bev_pool(self, geometry, features, *, return_5d=False):
            calls.append((geometry, features, return_5d))
            return expected

    Model.bev_pool._turbo_physai_returns_5d = True
    features = torch.zeros(1, 1, 1, 2, 3, 4)
    geometry = torch.zeros(1, 1, 1, 2, 3, 3)

    result = reference._bev_pool_5d(Model(), features, geometry)
    assert result is expected
    assert len(calls) == 1
    assert calls[0][0] is geometry
    assert calls[0][1] is features
    assert calls[0][2] is True


@pytest.mark.parametrize("installed_fix", [False, True])
@pytest.mark.parametrize("channels_last", [False, True])
def test_reference_pool_layout_values_and_gradients(monkeypatch, installed_fix, channels_last):
    from turbo_physai.optimizations.models.maptrv2 import reference

    batch, channels, depth, height, width = 2, 4, 2, 3, 4
    source = torch.arange(
        batch * channels * depth * height * width, dtype=torch.float32
    ).reshape(batch, channels, depth, height, width).requires_grad_()
    pooled = source.permute(0, 2, 3, 4, 1).contiguous() if channels_last else source
    model, features, geometry = _reference_pool_case(
        monkeypatch, pooled, channels, (height, width, depth), installed_fix
    )
    result = reference._bev_pool_5d(model, features, geometry)
    torch.testing.assert_close(result, source.detach().permute(0, 2, 3, 4, 1),
                               rtol=0, atol=0)
    if channels_last:
        assert result is pooled
    weights = torch.arange(1, result.numel() + 1, dtype=result.dtype).reshape(result.shape)
    (result * weights).sum().backward()
    torch.testing.assert_close(source.grad, weights.permute(0, 4, 1, 2, 3), rtol=0, atol=0)


@pytest.mark.parametrize("installed_fix", [False, True])
@pytest.mark.parametrize("shape,message", [
    ((1, 2, 2, 2, 2), "ambiguous BEV pooling layout"),
    ((1, 2, 2, 2), "invalid BEV pooling shape"),
    ((1, 2, 2, 2, 3), "invalid BEV pooling shape"),
])
def test_reference_pool_rejects_unidentifiable_layouts(monkeypatch, installed_fix, shape, message):
    from turbo_physai.optimizations.models.maptrv2 import reference

    pooled = torch.arange(float(np.prod(shape))).reshape(shape)
    model, features, geometry = _reference_pool_case(
        monkeypatch, pooled, 2, (2, 2, 2), installed_fix
    )
    with pytest.raises(RuntimeError, match=message):
        reference._bev_pool_5d(model, features, geometry)


def _reference_pool_case(monkeypatch, pooled, channels, grid, installed_fix):
    ops = types.ModuleType("mmdet3d.ops")
    ops.bev_pool = lambda *args: pooled
    monkeypatch.setitem(sys.modules, "mmdet3d.ops", ops)

    class Model:
        pass

    if installed_fix:
        from turbo_physai.optimizations.models.maptrv2 import bev_pool_fix

        monkeypatch.setattr(bev_pool_fix, "_bev_pool", ops.bev_pool)
        Model.bev_pool = bev_pool_fix.base_transform_bev_pool
    model = Model()
    model.C = channels
    model.nx = torch.tensor(grid)
    model.bx, model.dx = torch.zeros(3), torch.ones(3)
    batch = pooled.shape[0]
    features = torch.zeros(batch, 1, 1, 1, 1, channels)
    geometry = torch.zeros(batch, 1, 1, 1, 1, 3)
    return model, features, geometry


def test_vectorized_samples_accepts_vertex_sequences(monkeypatch):
    pytest.importorskip("shapely")

    from turbo_physai.optimizations.models.maptrv2 import pv_mask

    _install_lidar_instance_lines_stub(monkeypatch)

    model = types.SimpleNamespace(
        vec_classes=("boundary",),
        CLASS2LABEL={"boundary": 1},
        aux_seg={"use_aux_seg": False},
        sample_dist=1,
        num_samples=20,
        padding=False,
        fixed_num=-1,
        padding_value=-10000,
        patch_size=(100, 100),
    )
    result = pv_mask.gen_vectorized_samples(
        model, {"boundary": [[(0.0, 0.0), (1.0, 1.0)]]}
    )
    assert result["gt_vecs_label"] == [1]
    assert len(result["gt_vecs_pts_loc"].instances) == 1
    received = result["gt_vecs_pts_loc"].received
    assert (
        received["instance_line_list"]
        is result["gt_vecs_pts_loc"].instances
    )
    assert received["instance_labels"] == [1]
    assert received["sample_dist"] == 1
    assert received["num_samples"] == 20
    assert received["padding"] is False
    assert received["fixed_num"] == -1
    assert received["padding_value"] == -10000
    assert received["patch_size"] == (100, 100)


def test_vectorized_samples_aux_seg_call_contract(monkeypatch):
    pytest.importorskip("shapely")

    from turbo_physai.optimizations.models.maptrv2 import pv_mask

    _install_lidar_instance_lines_stub(monkeypatch)
    bev_calls = []
    pv_calls = []
    model = types.SimpleNamespace(
        vec_classes=("boundary",),
        CLASS2LABEL={"boundary": 1},
        aux_seg={
            "use_aux_seg": True,
            "seg_classes": 1,
            "bev_seg": True,
            "pv_seg": True,
            "pv_thickness": 2,
        },
        canvas_size=(100, 100),
        thickness=3,
        sample_dist=1,
        num_samples=20,
        padding=False,
        fixed_num=-1,
        padding_value=-10000,
        patch_size=(100, 100),
    )
    model.line_ego_to_mask = lambda *args, **kwargs: bev_calls.append(
        (args, kwargs)
    )
    model.line_ego_to_pvmask = lambda *args, **kwargs: pv_calls.append(
        (args, kwargs)
    )
    example = {
        "img_metas": types.SimpleNamespace(
            data={
                "pad_shape": [(64, 64), (64, 64)],
                "lidar2img": [
                    torch.eye(4).numpy(),
                    torch.eye(4).numpy(),
                ],
            }
        )
    }

    result = pv_mask.gen_vectorized_samples(
        model,
        {"boundary": [[(0.0, 0.0), (1.0, 1.0)]]},
        example=example,
    )

    assert len(bev_calls) == 1
    assert len(pv_calls) == 2
    assert bev_calls[0][1] == {"color": 1, "thickness": 3}
    assert all(call[1]["thickness"] == 2 for call in pv_calls)
    assert tuple(result["gt_semantic_mask"].shape) == (1, 100, 100)
    assert tuple(result["gt_pv_semantic_mask"].shape) == (2, 1, 2, 2)


def test_force_geometric_kernel_extension_is_idempotent(tmp_path):
    from turbo_physai.optimizations.models.maptrv2 import build

    setup_path = (
        tmp_path
        / "projects"
        / "mmdet3d_plugin"
        / "maptr"
        / "modules"
        / "ops"
        / "geometric_kernel_attn"
        / "setup.py"
    )
    setup_path.parent.mkdir(parents=True)
    setup_path.write_text(
        "    if torch.cuda.is_available() and CUDA_HOME is not None:\n",
        encoding="utf-8",
    )
    assert build.force_geometric_kernel_extension(tmp_path) is True
    assert build.force_geometric_kernel_extension(tmp_path) is False
    text = setup_path.read_text(encoding="utf-8")
    assert text.startswith("    if 1:\n")
    assert "# if torch.cuda.is_available()" in text


def test_bev_pool_compatibility_patch_is_explicit_and_idempotent(tmp_path):
    from turbo_physai.optimizations.models.maptrv2 import build

    bev_pool_path = (
        tmp_path
        / "mmdetection3d"
        / "mmdet3d"
        / "ops"
        / "bev_pool"
        / "bev_pool.py"
    )
    replacements = dict(
        build._build_compatibility_patches(tmp_path)
    )[bev_pool_path]
    bev_pool_path.parent.mkdir(parents=True)
    bev_pool_path.write_text(
        "def bev_pool(feats, coords, B, D, H, W):\n"
        "    x = pool(feats, coords, B, D, H, W)\n"
        "    x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    return x\n",
        encoding="utf-8",
    )

    assert build._replace_once(bev_pool_path, replacements) is True
    assert build._replace_once(bev_pool_path, replacements) is False
    text = bev_pool_path.read_text(encoding="utf-8")
    assert "    #x = x.permute(0, 4, 1, 2, 3).contiguous()" in text
    assert "\n    x = x.permute(0, 4, 1, 2, 3).contiguous()" not in text
