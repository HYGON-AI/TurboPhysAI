# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/datasets/nuscenes_offlinemap_dataset.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) 2022 Hust Vision Lab
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Vectorized perspective-view GT mask drawing for MapTRv2.

``VectorizedLocalMap.line_ego_to_pvmask`` walks every centerline with
``np.linspace(0, line_ego.length, 200)`` and then calls
``line_ego.interpolate(distance)`` once per sample.  Every sample is a shapely
round-trip that allocates a ``Point``, and the loop runs for each line and each
camera while the DataLoader workers build the auxiliary PV segmentation GT
(roughly 200 x lines x 6 calls per nuScenes sample).

The same polyline can be resampled with ``numpy`` alone: accumulate the segment
lengths of the vertices and interpolate on that axis.  Shapely interpolates
linearly inside the segment holding the requested arc length, and
``numpy.interp`` does the same on the cumulative-length axis, so the resampled
points -- and therefore the drawn mask -- match the official routine up to
float64 rounding.  ``perspective`` already drops the points behind the camera,
and :func:``project_points`` keeps that filter, so the visible geometry is
identical.

``VectorizedLocalMap.line_ego_to_mask`` paints the patch-level semantic mask
assembled by ``gen_vectorized_samples``.  The official body scales the line
with ``affinity.scale(..., origin=(0, 0))``, moves the result with a second
``affinity.affine_transform`` call, and rebuilds the vertex array through
``np.array(list(line_ego.coords), dtype=np.int32)[:, :2]``.
:func:``scale_translate_geom`` folds the scale and the ``canvas_size``/2
translation into one affine matrix, and :func:``line_ego_to_mask`` reads the
coordinates straight into ``numpy``.  Ego lines are two-dimensional, so
dropping the ``[:, :2]`` truncation is a no-op and the rasterized mask stays
pixel-identical.
"""

import numpy as np

#: Sample count used by the official ``line_ego_to_pvmask``.
PV_MASK_SAMPLES = 200


def sample_line_arc_length(line_ego, n_points=PV_MASK_SAMPLES):
    """Resample ``line_ego`` into ``n_points`` points spaced by arc length.

    Equivalent to ``[line_ego.interpolate(distance) for distance in
    np.linspace(0, line_ego.length, n_points)]`` without the shapely calls, and
    independent of the coordinate dimensionality of the source line.

    Shares its arc-length core with ``dataset_vectorization`` so the two
    replacements cannot drift apart.
    """

    from .dataset_vectorization import cumulative_lengths, resample_coordinates

    xs, ys = np.asarray(line_ego.xy, dtype=np.float64)
    coordinates, cumulative = cumulative_lengths(np.stack((xs, ys), axis=1))
    return resample_coordinates(
        coordinates,
        cumulative,
        np.linspace(0.0, cumulative[-1], n_points),
    )


def project_points(coords, proj_mat, z=0.0):
    """Project ego coordinates onto an image plane, dropping points behind it.

    Mirrors the module level ``perspective`` helper of the baseline dataset but
    takes the ``(N, 2)`` ego coordinates directly.
    """

    points = np.asarray(coords, dtype=np.float64)
    lidar_coords = np.concatenate(
        [
            points,
            np.full((points.shape[0], 1), z, dtype=np.float64),
            np.ones((points.shape[0], 1), dtype=np.float64),
        ],
        axis=1,
    ).transpose(1, 0)
    pix_coords = proj_mat @ lidar_coords
    valid = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    return pix_coords.transpose(1, 0)


def scale_translate_geom(geom, scale_x, scale_y, trans_x, trans_y,
                         origin=(0, 0)):
    """Scale ``geom`` and then translate it with a single affine matrix.

    ``affinity.scale(geom, sx, sy, origin=(x0, y0))`` uses the offsets
    ``x0 - sx * x0`` and ``y0 - sy * y0``, so scaling with it and then
    translating by ``(tx, ty)`` lands on::

        x' = sx * x + (tx + x0 - sx * x0)
        y' = sy * y + (ty + y0 - sy * y0)

    which is the single matrix built here.  Shapely therefore transforms the
    vertices once instead of twice.
    """

    from shapely import affinity

    x0, y0 = origin
    xoff = trans_x + x0 - scale_x * x0
    yoff = trans_y + y0 - scale_y * y0
    return affinity.affine_transform(
        geom, [scale_x, 0.0, 0.0, scale_y, xoff, yoff])


def line_ego_to_pvmask(self,
                       line_ego,
                       mask,
                       lidar2feat,
                       color=1,
                       thickness=1,
                       z=-1.6):
    """Drop-in replacement for ``VectorizedLocalMap.line_ego_to_pvmask``."""

    import cv2

    pix_coords = project_points(
        sample_line_arc_length(line_ego), lidar2feat, z=z)
    cv2.polylines(mask, np.int32([pix_coords]), False, color=color,
                  thickness=thickness)


def line_ego_to_mask(self,
                     line_ego,
                     mask,
                     color=1,
                     thickness=3):
    """Drop-in replacement for ``VectorizedLocalMap.line_ego_to_mask``.

    The baseline transforms the line twice (a scale around the origin and a
    separate translation) and rebuilds the vertices through a Python list;
    this version composes both steps and reads the coordinates directly.
    ``gen_vectorized_samples`` calls it for the patch-level semantic mask, so
    the saving repeats once per line of every label.  Ego lines are
    two-dimensional, hence dropping the ``[:, :2]`` truncation of the baseline
    changes nothing and the rasterized mask is pixel-identical.
    """

    import cv2

    trans_x = self.canvas_size[1] / 2
    trans_y = self.canvas_size[0] / 2
    line_ego = scale_translate_geom(
        line_ego, self.scale_x, self.scale_y, trans_x, trans_y)
    coords = np.asarray(line_ego.coords, dtype=np.int32)
    assert len(coords) >= 2

    cv2.polylines(mask, np.int32([coords]), False, color=color,
                  thickness=thickness)


def gen_vectorized_samples(
    self, map_annotation, example=None, feat_down_sample=32
):
    """Reference vectorization path without the redundant vertex copy."""

    from projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset import (
        LiDARInstanceLines,
    )
    from shapely.geometry import LineString

    vectors = []
    for vec_class in self.vec_classes:
        for instance in map_annotation[vec_class]:
            vectors.append(
                (
                    LineString(instance),
                    self.CLASS2LABEL.get(vec_class, -1),
                )
            )

    gt_labels = []
    gt_instance = []
    if self.aux_seg["use_aux_seg"]:
        if self.aux_seg["seg_classes"] == 1:
            if self.aux_seg["bev_seg"]:
                gt_semantic_mask = np.zeros(
                    (1, self.canvas_size[0], self.canvas_size[1]),
                    dtype=np.uint8,
                )
            else:
                gt_semantic_mask = None
            if self.aux_seg["pv_seg"]:
                num_cam = len(example["img_metas"].data["pad_shape"])
                img_shape = example["img_metas"].data["pad_shape"][0]
                gt_pv_semantic_mask = np.zeros(
                    (
                        num_cam,
                        1,
                        img_shape[0] // feat_down_sample,
                        img_shape[1] // feat_down_sample,
                    ),
                    dtype=np.uint8,
                )
                lidar2img = example["img_metas"].data["lidar2img"]
                scale_factor = np.eye(4)
                scale_factor[0, 0] *= 1 / 32
                scale_factor[1, 1] *= 1 / 32
                lidar2feat = [
                    scale_factor @ transform for transform in lidar2img
                ]
            else:
                gt_pv_semantic_mask = None
            for instance, instance_type in vectors:
                if instance_type == -1:
                    continue
                gt_instance.append(instance)
                gt_labels.append(instance_type)
                if instance.geom_type != "LineString":
                    print(instance.geom_type)
                    continue
                if self.aux_seg["bev_seg"]:
                    self.line_ego_to_mask(
                        instance,
                        gt_semantic_mask[0],
                        color=1,
                        thickness=self.thickness,
                    )
                if self.aux_seg["pv_seg"]:
                    for cam_index in range(num_cam):
                        self.line_ego_to_pvmask(
                            instance,
                            gt_pv_semantic_mask[cam_index][0],
                            lidar2feat[cam_index],
                            color=1,
                            thickness=self.aux_seg["pv_thickness"],
                        )
        else:
            if self.aux_seg["bev_seg"]:
                gt_semantic_mask = np.zeros(
                    (
                        len(self.vec_classes),
                        self.canvas_size[0],
                        self.canvas_size[1],
                    ),
                    dtype=np.uint8,
                )
            else:
                gt_semantic_mask = None
            if self.aux_seg["pv_seg"]:
                num_cam = len(example["img_metas"].data["pad_shape"])
                img_shape = example["img_metas"].data["pad_shape"][0]
                gt_pv_semantic_mask = np.zeros(
                    (
                        num_cam,
                        len(self.vec_classes),
                        img_shape[0] // feat_down_sample,
                        img_shape[1] // feat_down_sample,
                    ),
                    dtype=np.uint8,
                )
                lidar2img = example["img_metas"].data["lidar2img"]
                scale_factor = np.eye(4)
                scale_factor[0, 0] *= 1 / 32
                scale_factor[1, 1] *= 1 / 32
                lidar2feat = [
                    scale_factor @ transform for transform in lidar2img
                ]
            else:
                gt_pv_semantic_mask = None
            for instance, instance_type in vectors:
                if instance_type == -1:
                    continue
                gt_instance.append(instance)
                gt_labels.append(instance_type)
                if instance.geom_type != "LineString":
                    print(instance.geom_type)
                    continue
                if self.aux_seg["bev_seg"]:
                    self.line_ego_to_mask(
                        instance,
                        gt_semantic_mask[instance_type],
                        color=1,
                        thickness=self.thickness,
                    )
                if self.aux_seg["pv_seg"]:
                    for cam_index in range(num_cam):
                        self.line_ego_to_pvmask(
                            instance,
                            gt_pv_semantic_mask[cam_index][instance_type],
                            lidar2feat[cam_index],
                            color=1,
                            thickness=self.aux_seg["pv_thickness"],
                        )
    else:
        for instance, instance_type in vectors:
            if instance_type != -1:
                gt_instance.append(instance)
                gt_labels.append(instance_type)
        gt_semantic_mask = None
        gt_pv_semantic_mask = None

    gt_instance = LiDARInstanceLines(
        gt_instance,
        gt_labels,
        self.sample_dist,
        self.num_samples,
        self.padding,
        self.fixed_num,
        self.padding_value,
        patch_size=self.patch_size,
    )
    return {
        "gt_vecs_pts_loc": gt_instance,
        "gt_vecs_label": gt_labels,
        "gt_semantic_mask": gt_semantic_mask,
        "gt_pv_semantic_mask": gt_pv_semantic_mask,
    }


__all__ = [
    "PV_MASK_SAMPLES",
    "gen_vectorized_samples",
    "line_ego_to_mask",
    "line_ego_to_pvmask",
    "project_points",
    "sample_line_arc_length",
    "scale_translate_geom",
]
