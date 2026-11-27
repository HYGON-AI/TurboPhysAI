# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical consistency between the scipy focal/shift solve and the GPU port.

``moge.utils.geometry_torch.recover_focal_shift`` hands each sample to
``scipy.optimize.least_squares(method='lm')`` on the CPU.  The replacement in
``turbo_physai.optimizations.models.lingbot_vla.moge_focal_gpu`` solves the
same objective with a batched secant search on the point map's own device.

These tests build point maps from a known focal and shift, so both sides see
the same input and a change to either implementation shows up as a value
mismatch rather than a slow drift.  The replacement solves for shift over the
whole feasible range (``z + shift > 0``), so the cases include shifts far
outside the usual window.  ``focal`` is compared in relative terms because it
scales with the point map.
"""

from __future__ import annotations

import os

# Keep the replacement eager: the secant loop is compiled in production, and
# compiling it here would dominate the test runtime without changing values.
os.environ.setdefault("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")

import pytest


torch = pytest.importorskip("torch")

normalized_view_plane_uv = pytest.importorskip(
    "moge.utils.geometry_torch"
).normalized_view_plane_uv
upstream_recover = pytest.importorskip(
    "moge.utils.geometry_torch"
).recover_focal_shift
from turbo_physai.optimizations.models.lingbot_vla import moge_focal_gpu  # noqa: E402


pytestmark = pytest.mark.model_deps

SHIFT_ATOL = 2e-3
FOCAL_RTOL = 2e-4
HEIGHT = 32
WIDTH = 32

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _point_map(
    shift, focal, noise=0.0, seed=0, batch=1, height=HEIGHT, width=WIDTH, device="cpu"
):
    """Point map whose true focal/shift are known, so ``z + shift`` stays positive."""
    generator = torch.Generator().manual_seed(seed)
    uv = normalized_view_plane_uv(width, height, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, height),
        torch.linspace(0.0, 1.0, width),
        indexing="ij",
    )
    # Strictly positive depth, varying across the map so the fit is non-degenerate.
    # Keep ``z + shift`` positive for every shift under test.
    base = max(0.5, -shift + 0.5)
    depth = base + (0.5 + 0.5 * torch.sin(3.0 * xx) * torch.cos(2.0 * yy))
    xy = uv * (depth + shift).unsqueeze(-1) / focal
    if noise:
        xy = xy + noise * torch.randn(xy.shape, generator=generator)
    points = torch.cat((xy, depth.unsqueeze(-1)), dim=-1)
    return points.unsqueeze(0).repeat(batch, 1, 1, 1).to(device)


def _assert_matches_reference(points, mask=None):
    expected_focal, expected_shift = upstream_recover(points, mask)
    actual_focal, actual_shift = moge_focal_gpu.recover_focal_shift(points, mask)

    torch.testing.assert_close(
        actual_shift,
        expected_shift,
        atol=SHIFT_ATOL,
        rtol=0,
        msg="shift diverged from the scipy reference",
    )
    torch.testing.assert_close(
        actual_focal,
        expected_focal,
        atol=0,
        rtol=FOCAL_RTOL,
        msg="focal diverged from the scipy reference",
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift", [0.0, 0.1, 0.2, -0.3, 0.45, 2.0, -2.0, 5.0, -20.0])
@pytest.mark.parametrize("focal", [0.8, 1.5, 2.0])
def test_shift_and_focal_match_scipy_on_clean_points(device, shift, focal):
    _assert_matches_reference(_point_map(shift, focal, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("noise", [1e-3, 1e-2])
def test_shift_and_focal_match_scipy_with_noise(device, noise):
    _assert_matches_reference(
        _point_map(shift=-0.3, focal=1.5, noise=noise, seed=7, device=device)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_batched_samples_match_scipy(device):
    _assert_matches_reference(
        _point_map(shift=-0.2, focal=1.2, batch=4, seed=11, device=device)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_masked_points_match_scipy(device):
    points = _point_map(shift=-0.4, focal=1.5, seed=13, device=device)
    generator = torch.Generator().manual_seed(17)
    mask = (torch.rand(points.shape[:-1], generator=generator) > 0.2).to(
        points.device
    )
    _assert_matches_reference(points, mask)


@pytest.mark.parametrize("device", DEVICES)
def test_given_focal_delegates_to_the_scipy_path(device):
    """With a known focal the replacement returns the upstream result verbatim."""
    points = _point_map(shift=0.3, focal=1.5, device=device)
    focal = torch.full((1,), 1.5, device=device)

    expected_focal, expected_shift = upstream_recover(points, None, focal=focal)
    actual_focal, actual_shift = moge_focal_gpu.recover_focal_shift(
        points, None, focal=focal
    )

    assert torch.equal(actual_focal, expected_focal)
    assert torch.equal(actual_shift, expected_shift)
