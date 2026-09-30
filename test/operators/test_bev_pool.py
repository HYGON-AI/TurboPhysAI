# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.operators import _bev_pool as operator


def test_bev_pool_sorts_inputs_and_runs_autograd(monkeypatch):
    calls = []

    class FakeBevPool(torch.autograd.Function):
        @staticmethod
        def forward(ctx, feats, coords, ranks, batch, depth, height, width):
            calls.append(
                {
                    "feats": feats.clone(),
                    "coords": coords.clone(),
                    "ranks": ranks.clone(),
                    "shape": (batch, depth, height, width),
                }
            )
            return feats.new_zeros((batch, depth, height, width, feats.shape[1]))

        @staticmethod
        def backward(ctx, grad_output):
            calls.append(("backward", grad_output.shape))
            return torch.ones(3, 2), None, None, None, None, None, None

    monkeypatch.setattr(operator, "_NATIVE_BEV_POOL_AUTOGRAD", FakeBevPool)

    feats = torch.arange(6.0).reshape(3, 2).requires_grad_()
    coords = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    ranks = torch.tensor([20, 10, 10])

    output = operator.bev_pool(feats, coords, 1, 2, 3, 4, ranks=ranks)

    assert output.shape == (1, 2, 2, 3, 4)
    assert calls[0]["feats"].tolist() == [[2.0, 3.0], [4.0, 5.0], [0.0, 1.0]]
    assert calls[0]["coords"].tolist() == [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert calls[0]["ranks"].tolist() == [10, 10, 20]
    assert calls[0]["shape"] == (1, 2, 3, 4)

    output.sum().backward()

    assert feats.grad is not None
    assert feats.grad.tolist() == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    assert calls[-1] == ("backward", (1, 2, 3, 4, 2))


def test_bev_pool_prepare_and_geometry_delegate(monkeypatch):
    calls = []

    class Extension:
        @staticmethod
        def bev_pool_prepare(*args):
            calls.append(("prepare", args))
            return "prepared"

        @staticmethod
        def bev_pool_prepare_geometry(*args):
            calls.append(("geometry", args))
            return "geometry"

    monkeypatch.setattr(operator, "_ops", lambda: Extension)

    tensors = [torch.zeros(1) for _ in range(4)]
    assert operator.bev_pool_prepare(*tensors, 1, 2, 3, 4) == "prepared"
    assert [tensor.shape for tensor in calls[-1][1][:4]] == [(1,)] * 4
    assert calls[-1][1][4:] == (1, 2, 3, 4)

    geometry_tensors = [torch.zeros(1) for _ in range(10)]
    assert (
        operator.bev_pool_prepare_geometry(
            *geometry_tensors, 1, 2, 3, 4, boundary_eps=0.25
        )
        == "geometry"
    )
    assert [tensor.shape for tensor in calls[-1][1][:10]] == [(1,)] * 10
    assert calls[-1][1][10:] == (1, 2, 3, 4, 0.25)


def test_bev_pool_rejects_mismatched_feature_and_coordinate_rows():
    with pytest.raises(ValueError, match="equal feature/coord rows"):
        operator.bev_pool(
            torch.zeros(2, 3),
            torch.zeros(1, 4, dtype=torch.int32),
            1,
            2,
            3,
            4,
        )
