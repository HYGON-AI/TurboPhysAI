# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the Sparse4D target rewrite with the upstream sampler."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.model_deps


@pytest.fixture(scope="module")
def target_classes():
    registry = pytest.importorskip("mmdet.core.bbox.builder").BBOX_SAMPLERS
    pytest.importorskip("scipy.optimize")
    pytest.importorskip("torch_linear_assignment")

    from turbo_physai.engine.execution.replacements import default_handlers
    from turbo_physai.optimizations.models.sparse4d.catalog import DETECTION3D_TARGET

    with pytest.MonkeyPatch.context() as patch:
        sibling = Path(__file__).resolve().parents[3] / "Sparse4D"
        if importlib.util.find_spec("projects") is None and sibling.is_dir():
            patch.syspath_prepend(str(sibling))
        upstream = pytest.importorskip(
            "projects.mmdet3d_plugin.models.detection3d.target"
        ).SparseBox3DTarget

        (spec,) = DETECTION3D_TARGET.specs
        handler = default_handlers()[spec.mechanism]
        prepared = handler.prepare(spec, {})
        snapshot = handler.snapshot(prepared)
        try:
            handler.apply(prepared)
            optimized = registry.get(upstream.__name__)
            assert optimized is not upstream
            yield upstream, optimized
        finally:
            handler.restore(snapshot)


def _ground_truth(device="cpu"):
    boxes = torch.tensor(
        [
            [10, 1, 2, 2, 3, 4, 0.4, 0.1, 0.2],
            [20, 3, 4, 4, 2, 1, -0.7, 0.3, 0.4],
            [30, 5, 6, 3, 1, 2, 1.2, 0.5, 0.6],
            [40, 7, 8, 1, 4, 3, -1.5, 0.7, 0.8],
        ],
        dtype=torch.float32,
        device=device,
    )
    labels = torch.tensor([2, 0, 2, 1], device=device)
    ids = torch.tensor([11, 12, 13, 21], device=device)
    return tuple(list(value.split([3, 0, 1])) for value in (labels, boxes, ids))


def test_sparse4d_sample_matches_upstream_on_ragged_ground_truth(target_classes):
    upstream, optimized = target_classes
    options = dict(reg_weights=[1.0] * 10, cls_wise_reg_weights={2: [0.5] * 10})
    reference, sampler = upstream(**options), optimized(**options)
    labels, boxes, _ = _ground_truth()
    generator = torch.Generator().manual_seed(7)
    cls_pred = torch.randn(3, 4, 3, generator=generator)
    box_pred = torch.randn(3, 4, 10, generator=generator)

    encoded = reference.encode_reg_target(boxes)
    torch.testing.assert_close(sampler.encode_reg_target(boxes, box_pred.device), encoded)
    torch.testing.assert_close(
        sampler._cls_cost(cls_pred, labels), reference._cls_cost(cls_pred, labels)
    )
    weights = [torch.rand(box.shape, generator=generator) for box in encoded]
    torch.testing.assert_close(
        sampler._box_cost(box_pred, encoded, weights),
        reference._box_cost(box_pred, encoded, weights),
    )

    args = cls_pred, box_pred, labels, boxes
    actual = sampler.sample(*args)
    torch.testing.assert_close(actual, reference.sample(*args))
    cls_target, box_target, reg_weights = actual
    assert (cls_target != cls_pred.shape[-1]).sum(dim=1).tolist() == [3, 0, 1]
    torch.testing.assert_close(box_target[1], torch.zeros_like(box_target[1]))
    torch.testing.assert_close(reg_weights[cls_target == 2], torch.full((2, 10), 0.5))


def test_sparse4d_padded_box_encoding_zeroes_invalid_rows(target_classes):
    upstream, optimized = target_classes
    _, boxes, _ = _ground_truth()
    padded = torch.stack(
        [torch.nn.functional.pad(box, (0, 0, 0, 3 - len(box))) for box in boxes]
    )
    valid = torch.arange(3) < torch.tensor([len(box) for box in boxes])[:, None]
    expected = torch.zeros(3, 3, 10)
    expected[valid] = torch.cat(upstream().encode_reg_target(boxes))

    actual = optimized().encode_reg_target_fix(padded, valid)

    torch.testing.assert_close(actual, expected)


def test_sparse4d_fixed_box_cost_matches_upstream(target_classes):
    upstream, optimized = target_classes
    options = dict(box_weight=0.75, reg_weights=[0.0, 0.5, 2.0, 1.0, 1.5] * 2)
    generator = torch.Generator().manual_seed(17)
    predictions = torch.randn(3, 4, 10, generator=generator)
    targets = torch.randn(3, 2, 10, generator=generator)
    weights = torch.rand(3, 2, 10, generator=generator)

    actual = optimized(**options)._box_cost_fix(predictions, targets, weights)
    expected = torch.stack(upstream(**options)._box_cost(predictions, targets, weights))

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "device,max_dn_gt,add_neg_dn,with_ids,compiled",
    [
        pytest.param("cpu", 0, False, False, False, id="dynamic-positive-only"),
        pytest.param("cpu", 4, True, True, True, id="fixed-padding-compiled"),
        pytest.param("cpu", 2, True, True, False, id="fixed-truncation"),
        pytest.param(
            "cuda", 4, True, True, False, id="fixed-padding-hcu", marks=pytest.mark.hcu
        ),
    ],
)
@pytest.mark.filterwarnings("error:No CUDA backend available for linear assignment")
def test_sparse4d_dn_anchors_match_upstream(
    target_classes, device, max_dn_gt, add_neg_dn, with_ids, compiled
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA/HCU device required")

    upstream, optimized = target_classes
    options = dict(
        num_dn_groups=2,
        num_temp_dn_groups=int(with_ids),
        max_dn_gt=max_dn_gt,
        add_neg_dn=add_neg_dn,
        dn_noise_scale=0.05,
    )
    reference, sampler = upstream(**options), optimized(**options)
    if compiled:
        sampler._assign_and_scatter = torch.compile(
            sampler._assign_and_scatter, backend="eager", fullgraph=True
        )

    labels, boxes, ids = _ground_truth(device)
    capacity = max_dn_gt or max(map(len, labels))
    reference_labels, reference_boxes, reference_ids = labels, boxes, ids
    if max_dn_gt > max(map(len, labels)):
        # Give upstream the same capacity so both runs draw identical noise.
        pad = torch.nn.functional.pad
        reference_labels = [pad(x, (0, capacity - len(x)), value=-1) for x in labels]
        reference_boxes = [pad(x, (0, 0, 0, capacity - len(x)), value=1) for x in boxes]
        reference_ids = [pad(x, (0, capacity - len(x)), value=-1) for x in ids]

    devices = [torch.cuda.current_device()] if device == "cuda" else []
    results = []
    for target, args in (
        (reference, (reference_labels, reference_boxes, reference_ids)),
        (sampler, (labels, boxes, ids)),
    ):
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(13)
            if devices:
                torch.cuda.manual_seed(13)
            results.append(target.get_dn_anchors(*args))

    expected, actual = results
    torch.testing.assert_close(actual, expected)
    anchors, _, dn_labels, attn_mask, valid_mask, dn_ids = actual
    group_width = capacity * (2 if add_neg_dn else 1)
    assert anchors.shape == (3, 2 * group_width, 10)
    assert attn_mask.shape == (2 * group_width, 2 * group_width)
    assert not attn_mask[:group_width, :group_width].any()
    assert attn_mask[:group_width, group_width:].all()
    assert valid_mask.sum(dim=1).tolist() == [
        2 * min(len(label), capacity) * (2 if add_neg_dn else 1) for label in labels
    ]
    if with_ids:
        assert (dn_ids[dn_labels < 0] == -1).all()
    else:
        assert dn_ids is None


def test_sparse4d_empty_ground_truth_matches_upstream(target_classes):
    upstream, optimized = (target(num_dn_groups=2) for target in target_classes)
    labels = [torch.empty(0, dtype=torch.long)] * 2
    boxes = [torch.empty(0, 9)] * 2
    cls_pred = torch.zeros(2, 3, 4)
    box_pred = torch.zeros(2, 3, 10)

    torch.testing.assert_close(
        optimized.encode_reg_target(boxes), upstream.encode_reg_target(boxes)
    )
    torch.testing.assert_close(
        optimized.sample(cls_pred, box_pred, labels, boxes),
        upstream.sample(cls_pred, box_pred, labels, boxes),
    )
    assert upstream.get_dn_anchors(labels, boxes) is None
    assert optimized.get_dn_anchors(labels, boxes) is None


def test_sparse4d_temporal_dn_cache_and_update_match_upstream(target_classes, monkeypatch):
    group_order = torch.tensor([2, 0, 3, 1])

    def randperm(num_groups):
        assert num_groups == group_order.numel()
        return group_order

    monkeypatch.setattr(torch, "randperm", randperm)
    upstream, optimized = (
        target(num_dn_groups=4, num_temp_dn_groups=2) for target in target_classes
    )
    labels = torch.arange(16).reshape(2, 8)
    ids = labels + 100
    valid = labels.remainder(2).bool()
    features = torch.arange(48.0).reshape(2, 8, 3).requires_grad_()
    anchors = torch.arange(160.0).reshape(2, 8, 10).requires_grad_()
    for target in (upstream, optimized):
        target.cache_dn(features, anchors, labels, valid, ids)

    torch.testing.assert_close(optimized.dn_metas, upstream.dn_metas)
    assert optimized.dn_metas["dn_cls_target"][0].tolist() == [[2, 3], [6, 7]]
    assert not optimized.dn_metas["dn_instance_feature"].requires_grad
    assert not optimized.dn_metas["dn_anchor"].requires_grad

    current_ids = ids.clone()
    current_ids[0, :2] = torch.tensor([999, 102])
    args = (
        torch.nn.functional.pad(features, (0, 0, 1, 0)),
        torch.nn.functional.pad(anchors, (0, 0, 1, 0)),
        anchors.detach(),
        labels,
        valid,
        current_ids,
        1,
        torch.tensor([True, False]),
    )
    actual = optimized.update_dn(*args)
    torch.testing.assert_close(actual, upstream.update_dn(*args))
    torch.testing.assert_close(actual[2][0, 0], anchors[0, 1].detach())
    assert actual[3][0, :2].tolist() == [2, -1]
    torch.testing.assert_close(actual[3][1], labels[1])
