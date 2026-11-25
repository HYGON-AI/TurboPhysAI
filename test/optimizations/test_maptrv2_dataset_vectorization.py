# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the vectorized nuScenes offline-map resampling."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from turbo_physai.engine.definitions.registry import default_registry  # noqa: E402
import turbo_physai.optimizations.models.maptrv2.catalog as catalog  # noqa: E402
from turbo_physai.optimizations.models.maptrv2 import (  # noqa: E402
    dataset_vectorization,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
GROUP_ID = "maptrv2.dataset_vectorization"
TARGET = (
    "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
    "LiDARInstanceLines.shift_fixed_num_sampled_points_v2"
)


class _Line:
    """Minimal stand-in for a shapely ``LineString``."""

    def __init__(self, coordinates):
        self.coords = np.asarray(coordinates, dtype=np.float64)
        self.length = float(
            np.hypot(
                np.diff(self.coords[:, 0]),
                np.diff(self.coords[:, 1]),
            ).sum()
        )


def _legacy_interpolate(coordinates, distances):
    """Piecewise arc-length interpolation, written independently of NumPy interp."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    cumulative = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.hypot(
                    np.diff(coordinates[:, 0]),
                    np.diff(coordinates[:, 1]),
                )
            ),
        )
    )
    result = []
    for distance in distances:
        segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
        segment = min(max(segment, 0), coordinates.shape[0] - 2)
        start, end = cumulative[segment], cumulative[segment + 1]
        ratio = 0.0 if end == start else (distance - start) / (end - start)
        result.append(
            coordinates[segment]
            + ratio * (coordinates[segment + 1] - coordinates[segment])
        )
    return np.asarray(result, dtype=np.float64)


def _legacy_owner_tensor(owner):
    """Baseline ``shift_fixed_num_sampled_points_v2``, including the outer stack."""

    instances_list = []
    for index, instance in enumerate(owner.instance_list):
        coordinates = np.asarray(instance.coords, dtype=np.float64)
        distances = np.linspace(0, instance.length, owner.fixed_num)
        final_shift_num = owner.fixed_num - 1
        is_poly = np.array_equal(coordinates[0], coordinates[-1])

        if owner.instance_labels[index] == 3:
            shifts = [_legacy_interpolate(coordinates, distances)]
        elif is_poly:
            points = coordinates[:-1]
            shifts = []
            for shift in range(points.shape[0]):
                rolled = np.roll(points, shift, axis=0)
                rolled = np.concatenate((rolled, rolled[:1]), axis=0)
                shifts.append(_legacy_interpolate(rolled, distances))
        else:
            sampled = _legacy_interpolate(coordinates, distances)
            shifts = [sampled, np.flip(sampled, axis=0)]

        multi_shifts = np.stack(shifts, axis=0)
        if multi_shifts.shape[0] > final_shift_num:
            selected = np.random.choice(
                multi_shifts.shape[0], final_shift_num, replace=False
            )
            multi_shifts = multi_shifts[selected]

        tensor = torch.from_numpy(
            np.ascontiguousarray(multi_shifts, dtype=np.float32)
        )
        tensor[..., 0] = tensor[..., 0].clamp(min=-owner.max_x, max=owner.max_x)
        tensor[..., 1] = tensor[..., 1].clamp(min=-owner.max_y, max=owner.max_y)
        if tensor.shape[0] < final_shift_num:
            tensor = torch.cat(
                (
                    tensor,
                    tensor.new_full(
                        (final_shift_num - tensor.shape[0], owner.fixed_num, 2),
                        owner.padding_value,
                    ),
                ),
                dim=0,
            )
        instances_list.append(tensor)
    return torch.stack(instances_list, dim=0).to(dtype=torch.float32)


def _owner(coordinates_list, labels, fixed_num):
    return types.SimpleNamespace(
        instance_list=[_Line(coords) for coords in coordinates_list],
        instance_labels=list(labels),
        fixed_num=fixed_num,
        padding_value=-10000,
        max_x=30.0,
        max_y=60.0,
    )


CASES = (
    ("open polyline", [[(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]], [0], 5),
    ("closed polygon", [[(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)]], [0], 3),
    ("label 3 shortcut", [[(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]], [3], 6),
    ("repeated vertices", [[(0.0, 0.0), (0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]], [0], 5),
    ("short line is padded", [[(0.0, 0.0), (1.0, 0.0)]], [0], 7),
    (
        "several instances",
        [[(0.0, 0.0), (10.0, 0.0)], [(0.0, 0.0), (2.0, 2.0)], [(-5.0, -5.0), (5.0, 5.0)]],
        [0, 1, 3],
        4,
    ),
    (
        "polygon wider than the shift budget",
        [[(float(i), float((i * 3) % 7)) for i in range(12)] + [(0.0, 0.0)]],
        [0],
        3,
    ),
)


@pytest.mark.parametrize("name,coordinates_list,labels,fixed_num", CASES)
def test_shift_v2_matches_legacy_implementation(
    name, coordinates_list, labels, fixed_num
):
    np.random.seed(12345)
    expected = _legacy_owner_tensor(_owner(coordinates_list, labels, fixed_num))
    np.random.seed(12345)
    actual = dataset_vectorization.shift_fixed_num_sampled_points_v2.fget(
        _owner(coordinates_list, labels, fixed_num)
    )
    assert actual.shape == expected.shape, name
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_interpolation_matches_piecewise_legacy_result():
    # Total length is 2 + 3 + 3 = 8; sampling stays inside [0, length] exactly
    # like ``np.linspace(0, instance.length, n)`` does upstream.
    coordinates = [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (5.0, 3.0)]
    distances = np.linspace(0.0, 8.0, 17)
    actual = dataset_vectorization.interpolate_coordinates(coordinates, distances)
    expected = _legacy_interpolate(coordinates, distances)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_distances_past_the_end_clamp_to_the_last_vertex():
    """``np.interp`` clamps, which is what shapely ``interpolate`` does too.

    The upstream callers never ask for more than ``line.length``, but the two
    must still agree on the boundary rather than extrapolating.
    """

    coordinates = [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0)]
    actual = dataset_vectorization.interpolate_coordinates(
        coordinates, np.array([6.0, 10.0])
    )
    np.testing.assert_allclose(actual, [[2.0, 3.0], [2.0, 3.0]], rtol=0.0, atol=0.0)


def test_repeated_vertices_do_not_change_the_result():
    """Zero-length segments are dropped, which cannot move any sample."""

    clean = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    padded = [(0.0, 0.0), (0.0, 0.0), (3.0, 0.0), (3.0, 0.0), (3.0, 4.0)]
    distances = np.linspace(0.0, 7.0, 9)
    np.testing.assert_allclose(
        dataset_vectorization.interpolate_coordinates(clean, distances),
        dataset_vectorization.interpolate_coordinates(padded, distances),
        rtol=0.0,
        atol=0.0,
    )


def test_pv_mask_shares_the_same_arc_length_core():
    """Both replacements must resample identically for the same polyline."""

    from turbo_physai.optimizations.models.maptrv2 import pv_mask

    coordinates = np.asarray(
        [(0.0, 0.0), (2.0, 1.0), (4.0, -1.0), (7.0, 2.0)], dtype=np.float64
    )

    class _Shapely:
        xy = (coordinates[:, 0], coordinates[:, 1])

    shared = dataset_vectorization.interpolate_coordinates(
        coordinates, np.linspace(0.0, float(np.hypot(np.diff(coordinates[:, 0]), np.diff(coordinates[:, 1])).sum()), 200)
    )
    np.testing.assert_allclose(
        pv_mask.sample_line_arc_length(_Shapely()), shared, rtol=0.0, atol=0.0
    )


def test_group_targets_the_property_with_a_property_replacement():
    group = default_registry.get_group(GROUP_ID)
    assert group is not None
    assert len(group.members) == 1
    spec = default_registry.get_spec(group.members[0])
    assert spec.target == TARGET
    assert spec.replacement == (
        "turbo_physai.optimizations.models.maptrv2."
        "dataset_vectorization.shift_fixed_num_sampled_points_v2"
    )
    assert spec.runtime_condition is None


def test_only_the_v2_property_is_exported():
    """The v0/v1/v3/v4 properties are dead under ``gt_shift_pts_pattern='v2'``."""

    assert dataset_vectorization.__all__ == [
        "cumulative_lengths",
        "interpolate_coordinates",
        "resample_coordinates",
        "shift_fixed_num_sampled_points_v2",
    ]
    assert isinstance(
        dataset_vectorization.shift_fixed_num_sampled_points_v2, property
    )


def test_matches_the_clean_checkout_implementation():
    """Compare bit-for-bit against the real baseline property when available."""

    checkout = REPO_ROOT.parent / "model" / "MapTrv2"
    if not checkout.is_dir():
        pytest.skip("sibling clean MapTrv2 checkout is unavailable")
    pytest.importorskip("shapely")
    script = r'''
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
from shapely.geometry import LineString

# A missing dependency is "cannot check here" (exit 3 -> skip); anything the
# comparison itself raises must keep the non-zero status so the test fails.
try:
    from projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset import (
        LiDARInstanceLines,
    )
    from turbo_physai.optimizations.models.maptrv2 import (
        dataset_vectorization as dv,
    )
except Exception as exc:
    print("UNAVAILABLE: %s: %s" % (type(exc).__name__, exc))
    raise SystemExit(3)

baseline = LiDARInstanceLines.__dict__[
    "shift_fixed_num_sampled_points_v2"
].fget
replacement = dv.shift_fixed_num_sampled_points_v2.fget
assert baseline is not replacement

CASES = [
    ([[(0, 0), (10, 0)]], [0], 5),
    ([[(0, 0), (3, 0), (3, 4)]], [0], 5),
    ([[(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]], [0], 3),
    ([[(i, (i * 3) % 7) for i in range(12)] + [(0, 0)]], [0], 3),
    ([[(0, 0), (3, 0), (3, 4)]], [3], 6),
    ([[(0, 0), (10, 0)], [(0, 0), (2, 2)], [(-5, -5), (5, 5)]], [0, 1, 3], 4),
    ([[(0, 0), (0, 0), (3, 0), (3, 4)]], [0], 5),
    ([[(0, 0), (1, 0)]], [0], 7),
]

rng = np.random.default_rng(7)
for _ in range(30):
    count = int(rng.integers(2, 15))
    coordinates = np.cumsum(rng.normal(scale=3.0, size=(count, 2)), axis=0)
    closed = rng.random() < 0.4
    if closed:
        coordinates = np.vstack([coordinates, coordinates[0]])
    CASES.append(
        (
            [coordinates.tolist()],
            [int(rng.choice([0, 1, 2, 3]))],
            int(rng.integers(2, 12)),
        )
    )


def build(coordinates_list, labels, fixed_num):
    return LiDARInstanceLines(
        [LineString(item) for item in coordinates_list],
        labels,
        patch_size=(60.0, 120.0),
        fixed_num=fixed_num,
        padding_value=-10000.0,
    )


def evaluate(owner, function):
    np.random.seed(20240929)
    return function(owner)


for index, (coordinates_list, labels, fixed_num) in enumerate(CASES):
    expected = evaluate(build(coordinates_list, labels, fixed_num), baseline)
    actual = evaluate(build(coordinates_list, labels, fixed_num), replacement)
    assert actual.shape == expected.shape, f"case {index}: shape"
    assert torch.equal(actual, expected), f"case {index}: values"
print("EQUIVALENT")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(checkout),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(checkout), str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
            ),
        },
    )
    if completed.returncode == 3:
        pytest.skip(completed.stdout.strip() or "baseline checkout unavailable")
    assert completed.returncode == 0, (
        f"vectorized v2 differs from the baseline property:\n{completed.stderr}"
    )
    assert "EQUIVALENT" in completed.stdout


def test_catalog_exports_the_group():
    assert "DATASET_VECTORIZATION" in catalog.__all__
    assert catalog.DATASET_VECTORIZATION.group_id == GROUP_ID
