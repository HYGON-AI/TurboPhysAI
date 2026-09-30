# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.operators import voxelization as operator


def _require_hcu():
    if not torch.cuda.is_available():
        pytest.skip("a real HCU device is required")
    import turbo_physai._C  # noqa: F401


def test_dynamic_voxelize_allocates_coordinates(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def dynamic_voxelize(points, coords, voxel_size, coors_range, ndim):
            calls.append((points, coords, voxel_size, coors_range, ndim))
            coords.fill_(3)

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    points = torch.zeros((2, 4), dtype=torch.float32)
    coords = operator.dynamic_voxelize(points, [1, 1, 1], range(6))

    assert coords.shape == (2, 3)
    assert coords.dtype == torch.int32
    assert torch.equal(coords, torch.full_like(coords, 3))
    assert calls[0][2] == [1, 1, 1]
    assert calls[0][3] == range(6)
    assert calls[0][4] == 3


def test_hard_voxelize_allocates_buffers_and_slices_count(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def hard_voxelize(points, voxels, coords, counts, *args):
            calls.append((points, voxels, coords, counts, args))
            voxels[0, 0] = points[0]
            coords[0] = torch.tensor([1, 2, 3], dtype=torch.int32)
            counts[0] = 1
            return 1

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    points = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    voxels, coords, counts = operator.hard_voxelize(
        points,
        [1, 1, 1],
        [0, 0, 0, 4, 4, 4],
        max_points=2,
        max_voxels=4,
    )

    assert voxels.shape == (1, 2, 4)
    assert coords.shape == (1, 3)
    assert counts.shape == (1,)
    assert voxels[0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert coords.tolist() == [[1, 2, 3]]
    assert calls[0][4] == ([1, 1, 1], [0, 0, 0, 4, 4, 4], 2, 4, 3, True)


def test_voxelize_frontend_uses_dynamic_and_hard_paths(monkeypatch):
    monkeypatch.setattr(operator, "dynamic_voxelize", lambda *args: ("dynamic", args))
    monkeypatch.setattr(operator, "hard_voxelize", lambda *args: ("hard", args))

    assert operator.voxelize("points", "size", "range", max_points=-1)[0] == "dynamic"
    assert operator.voxelize("points", "size", "range", max_points=35)[0] == "hard"


def test_dynamic_scatter_autograd_uses_forward_and_backward(monkeypatch):
    calls = []

    def forward(feats, coors, reduce_type):
        calls.append(("forward", reduce_type))
        reduced = feats[:1].clone()
        return (
            reduced,
            coors[:1].clone(),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )

    def backward(
        grad_feats,
        grad_reduced_feats,
        feats,
        reduced_feats,
        coors_idx,
        reduce_count,
        reduce_type,
    ):
        calls.append(("backward", reduce_type))
        grad_feats.copy_(grad_reduced_feats.expand_as(grad_feats))

    monkeypatch.setattr(operator, "_dynamic_point_to_voxel_forward", forward)
    monkeypatch.setattr(operator, "_dynamic_point_to_voxel_backward", backward)

    feats = torch.ones(3, 2, requires_grad=True)
    coors = torch.zeros(3, 3, dtype=torch.int32)
    reduced, out_coors = operator.dynamic_scatter(feats, coors, "mean")

    reduced.sum().backward()

    assert calls == [("forward", "mean"), ("backward", "mean")]
    torch.testing.assert_close(feats.grad, torch.ones_like(feats))
    assert out_coors.requires_grad is False


def test_dynamic_scatter_rejects_unknown_reduce_type():
    with pytest.raises(ValueError, match="unsupported reduce type"):
        operator.dynamic_scatter(
            torch.zeros(1, 1),
            torch.zeros(1, 3, dtype=torch.int32),
            "invalid",
        )


@pytest.mark.hcu
@pytest.mark.parametrize("reduce_type", ["max", "sum", "mean"])
def test_dynamic_scatter_matches_reference_forward_and_backward(reduce_type):
    _require_hcu()
    torch.manual_seed(1234)
    device = "cuda"

    feats = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [5.0, 0.0, 2.0],
            [2.0, 4.0, 2.0],
            [9.0, 9.0, 9.0],
            [0.0, 6.0, 4.0],
        ],
        device=device,
        requires_grad=True,
    )
    coors = torch.tensor(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [-1, -1, -1],
            [2, 0, 0],
        ],
        dtype=torch.int32,
        device=device,
    )

    valid = (coors >= 0).all(dim=1)
    expected_coors, inverse, counts = torch.unique(
        coors[valid],
        dim=0,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    expected = torch.empty(
        expected_coors.shape[0], feats.shape[1], device=device
    )
    for group, count in enumerate(counts.tolist()):
        members = valid_indices[inverse == group]
        grouped = feats.detach()[members]
        if reduce_type == "max":
            expected[group] = grouped.max(dim=0).values
        elif reduce_type == "sum":
            expected[group] = grouped.sum(dim=0)
        else:
            expected[group] = grouped.sum(dim=0) / count

    actual, actual_coors = operator.dynamic_scatter(feats, coors, reduce_type)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_coors, expected_coors)

    grad_reduced = torch.randn_like(expected)
    actual.backward(grad_reduced)

    expected_feature_grad = torch.zeros_like(feats)
    for group, count in enumerate(counts.tolist()):
        members = valid_indices[inverse == group]
        if reduce_type in {"sum", "mean"}:
            factor = 1.0 / count if reduce_type == "mean" else 1.0
            expected_feature_grad[members] = expected_feature_grad[members] + (
                grad_reduced[group] * factor
            )
        else:
            for channel in range(feats.shape[1]):
                channel_values = feats.detach()[members, channel]
                matched = members[channel_values == expected[group, channel]]
                expected_feature_grad[int(matched.min()), channel] += grad_reduced[
                    group, channel
                ]

    torch.testing.assert_close(
        feats.grad, expected_feature_grad, rtol=1e-5, atol=1e-5
    )
