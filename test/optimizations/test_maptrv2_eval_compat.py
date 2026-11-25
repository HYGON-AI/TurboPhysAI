# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the MapTRv2 validation compatibility fixes.

Covers two validation-time failures on newer dependency stacks:

* ``maptrv2.training`` -- ``MMDistributedDataParallel._run_ddp_forward`` reads
  ``_use_replicated_tensor_module``, which newer Torch DDP does not define.
* ``maptrv2.eval_compat`` -- Shapely 2 ``STRtree.query`` returns indices, while
  the baseline ``custom_polyline_score`` treats them as geometries.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


PACKAGE = "turbo_physai.optimizations.models.maptrv2"
CHAMFER_TARGET = (
    "projects.mmdet3d_plugin.datasets.map_utils.tpfp_chamfer.custom_polyline_score"
)
TPFP_ALIAS = "projects.mmdet3d_plugin.datasets.map_utils.tpfp.custom_polyline_score"
DDP_FORWARD_TARGET = (
    "mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward"
)


def _replacements(group):
    return {item.target: item for item in group.replacements}


def test_training_group_wraps_ddp_forward():
    catalog = importlib.import_module(f"{PACKAGE}.catalog")
    assert catalog.TRAINING.definition.group_id == "maptrv2.training"

    item = _replacements(catalog.TRAINING)[DDP_FORWARD_TARGET]
    assert item.replacement == f"{PACKAGE}.training.ddp_forward_compat_wrapper"


def test_ddp_forward_wrapper_defaults_missing_attribute_to_false():
    training = importlib.import_module(f"{PACKAGE}.training")

    class Ddp:
        def forward(self):
            return self._use_replicated_tensor_module

    wrapped = training.ddp_forward_compat_wrapper(Ddp.forward, {})
    assert wrapped(Ddp()) is False


def test_ddp_forward_wrapper_keeps_an_existing_attribute():
    training = importlib.import_module(f"{PACKAGE}.training")

    class Ddp:
        _use_replicated_tensor_module = True

        def forward(self):
            return self._use_replicated_tensor_module

    wrapped = training.ddp_forward_compat_wrapper(Ddp.forward, {})
    assert wrapped(Ddp()) is True


def test_eval_compat_group_replaces_the_function_and_its_importer():
    catalog = importlib.import_module(f"{PACKAGE}.catalog")
    assert catalog.EVAL_COMPAT.definition.group_id == "maptrv2.eval_compat"

    item = _replacements(catalog.EVAL_COMPAT)[CHAMFER_TARGET]
    assert item.replacement == f"{PACKAGE}.eval_compat.custom_polyline_score"
    assert TPFP_ALIAS in item.aliases
    assert item.runtime_condition == (
        f"{PACKAGE}.compat.shapely_query_returns_indices"
    )


def test_shapely_condition_honours_the_kill_switch(monkeypatch):
    compat = importlib.import_module(f"{PACKAGE}.compat")

    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_EVAL_COMPAT", raising=False)
    shapely = pytest.importorskip("shapely")
    expected = int(shapely.__version__.split(".", 1)[0]) >= 2
    assert compat.shapely_query_returns_indices() is expected

    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_EVAL_COMPAT", "1")
    assert compat.shapely_query_returns_indices() is False


def _brute_force(pred, gt, metric):
    """Baseline arithmetic without the spatial index."""

    from scipy.spatial import distance
    from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString

    def buffered(line):
        return LineString(line).buffer(
            1.0, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)

    preds = [buffered(line) for line in pred]
    gts = [buffered(line) for line in gt]
    if metric == "chamfer":
        matrix = np.full((len(preds), len(gts)), -100.0)
    else:
        matrix = np.zeros((len(preds), len(gts)))
    for j, gt_poly in enumerate(gts):
        for i, pred_poly in enumerate(preds):
            if not pred_poly.intersects(gt_poly):
                continue
            if metric == "chamfer":
                dist = distance.cdist(pred[i], gt[j])
                matrix[i, j] = -(dist.min(-1).mean() + dist.min(-2).mean()) / 2
            else:
                matrix[i, j] = (
                    pred_poly.intersection(gt_poly).area
                    / pred_poly.union(gt_poly).area
                )
    return matrix


@pytest.mark.parametrize("metric", ["chamfer", "iou"])
def test_custom_polyline_score_matches_the_unindexed_baseline(metric):
    pytest.importorskip("scipy")
    shapely = pytest.importorskip("shapely")
    if int(shapely.__version__.split(".", 1)[0]) < 2:
        pytest.skip("replacement targets the Shapely 2 STRtree contract")
    eval_compat = importlib.import_module(f"{PACKAGE}.eval_compat")

    # Short polylines (random start plus small steps) scattered over a wide
    # area, so both intersecting and non-intersecting pairs occur.
    rng = np.random.default_rng(0)

    def polylines(count):
        start = rng.uniform(-30, 30, (count, 1, 2))
        steps = rng.uniform(-0.4, 0.4, (count, 19, 2))
        return start + np.concatenate(
            (np.zeros((count, 1, 2)), np.cumsum(steps, axis=1)), axis=1)

    pred = polylines(16)
    gt = np.concatenate((pred[:5] + 0.3, polylines(6)))

    result = eval_compat.custom_polyline_score(pred, gt, metric=metric)
    expected = _brute_force(pred, gt, metric)

    miss = -100.0 if metric == "chamfer" else 0.0
    hits = int((expected != miss).sum())
    assert 0 < hits < expected.size
    assert result.shape == (16, 11)
    np.testing.assert_allclose(result, expected)


def test_custom_polyline_score_rejects_unknown_metric():
    pytest.importorskip("shapely")
    eval_compat = importlib.import_module(f"{PACKAGE}.eval_compat")

    lines = np.zeros((1, 3, 2))
    with pytest.raises(NotImplementedError):
        eval_compat.custom_polyline_score(lines, lines, metric="other")


INIT_DIST_TARGET = "mmcv.runner.dist_utils.init_dist"


def test_training_group_wraps_init_dist_with_its_importer_alias():
    catalog = importlib.import_module(f"{PACKAGE}.catalog")

    item = _replacements(catalog.TRAINING)[INIT_DIST_TARGET]
    assert item.replacement == f"{PACKAGE}.training.init_dist_fork_wrapper"
    assert "mmcv.runner.init_dist" in item.aliases


@pytest.fixture
def start_method(monkeypatch):
    """Fake ``torch.multiprocessing`` start-method state."""

    torch = pytest.importorskip("torch")
    state = {"method": None}

    monkeypatch.setattr(
        torch.multiprocessing, "get_start_method",
        lambda allow_none=False: state["method"])
    monkeypatch.setattr(
        torch.multiprocessing, "set_start_method",
        lambda method, force=False: state.update(method=method))
    monkeypatch.delenv("TURBO_PHYSAI_FORK_START_METHOD", raising=False)
    monkeypatch.delenv("TURBO_PHYSAI_DATALOADER_START_METHOD", raising=False)
    return state



def test_init_dist_wrapper_sets_fork_before_the_original_runs(start_method):
    training = importlib.import_module(f"{PACKAGE}.training")
    seen = {}

    def original(*args, **kwargs):
        seen["method"] = start_method["method"]
        seen["call"] = (args, kwargs)
        return "initialised"

    wrapped = training.init_dist_fork_wrapper(original, {})
    assert wrapped("pytorch", backend="nccl") == "initialised"
    assert seen["method"] == "fork"
    assert seen["call"] == (("pytorch",), {"backend": "nccl"})


def test_init_dist_wrapper_keeps_an_existing_start_method(start_method):
    training = importlib.import_module(f"{PACKAGE}.training")
    start_method["method"] = "spawn"

    training.init_dist_fork_wrapper(lambda *a, **k: None, {})("pytorch")
    assert start_method["method"] == "spawn"


def test_init_dist_wrapper_honours_the_switch(start_method, monkeypatch):
    training = importlib.import_module(f"{PACKAGE}.training")
    monkeypatch.setenv("TURBO_PHYSAI_FORK_START_METHOD", "0")

    training.init_dist_fork_wrapper(lambda *a, **k: None, {})("pytorch")
    assert start_method["method"] is None
