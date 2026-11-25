# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/datasets/nuscenes_offlinemap_dataset.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) 2022 Hust Vision Lab
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Vectorized ``LiDARInstanceLines.shift_fixed_num_sampled_points_v2``.

The baseline resamples every instance with a Python loop around
``LineString.interpolate``: each sample allocates a GEOS ``Point`` and copies its
coordinates back into Python containers, and a closed polygon repeats that once
per vertex shift.  The replacement measures the polyline once and evaluates all
sample positions with two ``np.interp`` calls.

Only ``shift_fixed_num_sampled_points_v2`` is replaced.  The sibling properties
(``fixed_num_sampled_points``, ``fixed_num_sampled_points_ambiguity``,
``shift_fixed_num_sampled_points``) are reached only through the ``v0``/``v1``/
``v3``/``v4`` patterns of ``gt_shift_pts_pattern``; the validated MapTRv2 config
selects ``v2``, so declaring them would add targets that never run.

Interior sample points repeat endpoints because ``np.linspace(0, length, n)``
includes both ends every time, just like the baseline.
"""

from __future__ import annotations

import numpy as np


def cumulative_lengths(coords):
    """Return ``(coordinates, cumulative)`` for an ``(N, 2)`` point sequence.

    ``cumulative`` is the arc length at each vertex, starting at zero, so it can
    be used directly as the ``xp`` argument of ``np.interp``.
    """

    coordinates = np.asarray(coords, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("line coordinates must have shape (N, 2)")
    if coordinates.shape[0] == 0:
        raise ValueError("line coordinates must not be empty")
    coordinates = coordinates[:, :2]
    segment_lengths = np.hypot(
        np.diff(coordinates[:, 0]),
        np.diff(coordinates[:, 1]),
    )
    cumulative = np.concatenate(
        (np.zeros(1, dtype=np.float64), np.cumsum(segment_lengths))
    )
    return coordinates, cumulative


def resample_coordinates(coordinates, cumulative, distances):
    """Sample ``distances`` along an already measured coordinate sequence."""

    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("distances must be one-dimensional")
    if distances.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # np.interp requires increasing xp.  A repeated vertex has zero arc length
    # between itself and its predecessor, which also makes the two coordinates
    # equal, so dropping the duplicate is numerically neutral.
    keep = np.concatenate(
        (np.ones(1, dtype=bool), np.diff(cumulative) > 0)
    )
    coordinates = coordinates[keep]
    cumulative = cumulative[keep]
    if cumulative.size == 1:
        return np.repeat(coordinates, distances.size, axis=0)

    return np.stack(
        (
            np.interp(distances, cumulative, coordinates[:, 0]),
            np.interp(distances, cumulative, coordinates[:, 1]),
        ),
        axis=1,
    )


def interpolate_coordinates(coords, distances):
    """Return points sampled at ``distances`` along a coordinate sequence."""

    coordinates, cumulative = cumulative_lengths(coords)
    return resample_coordinates(coordinates, cumulative, distances)


def _polygon_shift_coordinates(poly_pts):
    """Vertices of every cyclic shift of an open copy of a closed polygon.

    Shift ``i`` is the baseline's ``np.roll(points, i)`` followed by re-appending
    the first vertex, produced here as one gather.
    """

    points = poly_pts[:-1]
    count = points.shape[0]
    if count == 0:
        return np.empty((0, 0, 2), dtype=np.float64)
    offsets = np.arange(count, dtype=np.int64)
    indices = (offsets[None, :] - offsets[:, None]) % count
    shifted = points[indices]
    return np.concatenate((shifted, shifted[:, :1, :]), axis=1)


def _to_float_tensor(values):
    import torch

    array = np.ascontiguousarray(values, dtype=np.float32)
    return torch.from_numpy(array)


def _clamp_points(tensor, owner):
    tensor[..., 0] = tensor[..., 0].clamp(min=-owner.max_x, max=owner.max_x)
    tensor[..., 1] = tensor[..., 1].clamp(min=-owner.max_y, max=owner.max_y)
    return tensor


def _shifted_samples(instance, instance_label, fixed_num):
    """Return the ``(num_shifts, fixed_num, 2)`` samples for one instance."""

    distances = np.linspace(0, instance.length, fixed_num)
    poly_pts = np.asarray(instance.coords, dtype=np.float64)
    is_poly = np.array_equal(poly_pts[0], poly_pts[-1])

    if instance_label == 3:
        return [interpolate_coordinates(poly_pts, distances)]
    if is_poly:
        return [
            interpolate_coordinates(shifted, distances)
            for shifted in _polygon_shift_coordinates(poly_pts)
        ]
    sampled_points = interpolate_coordinates(poly_pts, distances)
    return [sampled_points, np.flip(sampled_points, axis=0)]


def _shift_fixed_num_sampled_points_v2(self):
    """Vectorized equivalent of the baseline property, same output layout."""

    import torch

    assert len(self.instance_list) != 0
    final_shift_num = self.fixed_num - 1
    instances_list = []
    for index, instance in enumerate(self.instance_list):
        multi_shifts = np.stack(
            _shifted_samples(instance, self.instance_labels[index], self.fixed_num),
            axis=0,
        )
        if multi_shifts.shape[0] > final_shift_num:
            selected = np.random.choice(
                multi_shifts.shape[0], final_shift_num, replace=False
            )
            multi_shifts = multi_shifts[selected]

        tensor = _to_float_tensor(multi_shifts)
        _clamp_points(tensor, self)
        if tensor.shape[0] < final_shift_num:
            tensor = torch.cat(
                (
                    tensor,
                    tensor.new_full(
                        (final_shift_num - tensor.shape[0], self.fixed_num, 2),
                        self.padding_value,
                    ),
                ),
                dim=0,
            )
        instances_list.append(tensor)

    return torch.stack(instances_list, dim=0).to(dtype=torch.float32)


shift_fixed_num_sampled_points_v2 = property(_shift_fixed_num_sampled_points_v2)


__all__ = [
    "cumulative_lengths",
    "interpolate_coordinates",
    "resample_coordinates",
    "shift_fixed_num_sampled_points_v2",
]
