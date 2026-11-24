# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/datasets/map_utils/tpfp_chamfer.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) 2022 Hust Vision Lab
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Shapely 2 compatible ``custom_polyline_score`` for the chamfer evaluation.

The baseline iterates ``STRtree.query(geometry)`` and treats each item as a
geometry (``o.intersects(...)``, ``id(o)``). That is the Shapely 1.x contract.
Shapely 2 returns an integer index array into the geometries the tree was built
from, so the baseline raises ``AttributeError: 'numpy.int64' object has no
attribute 'intersects'`` during validation.

The replacement keeps the baseline arithmetic unchanged and only resolves each
index back to the matching prediction polygon. The ``id()`` lookup table is no
longer needed because the index already is the prediction id.
"""

from __future__ import annotations

import numpy as np


def custom_polyline_score(pred_lines, gt_lines, linewidth=1., metric='chamfer'):
    """Pairwise score between predicted and ground-truth polylines.

    pred_lines: ``(num_preds, npts, 2)``; gt_lines: ``(num_gts, npts, 2)``.
    Returns a ``(num_preds, num_gts)`` matrix: negative mean chamfer distance
    for ``metric='chamfer'`` (``-100`` where the buffered lines do not
    intersect) or IoU of the buffered lines for ``metric='iou'``.
    """

    from scipy.spatial import distance
    from shapely.geometry import CAP_STYLE, JOIN_STYLE, LineString
    from shapely.strtree import STRtree

    if metric == 'iou':
        linewidth = 1.0
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)

    pred_lines_shapely = [
        LineString(line).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
        for line in pred_lines
    ]
    gt_lines_shapely = [
        LineString(line).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre)
        for line in gt_lines
    ]

    tree = STRtree(pred_lines_shapely)

    if metric == 'chamfer':
        iou_matrix = np.full((num_preds, num_gts), -100.)
    elif metric == 'iou':
        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    else:
        raise NotImplementedError

    for i, pline in enumerate(gt_lines_shapely):
        # Shapely >= 2.0: ``query`` returns indices into ``pred_lines_shapely``.
        for pred_id in tree.query(pline):
            pred_id = int(pred_id)
            o = pred_lines_shapely[pred_id]
            if not o.intersects(pline):
                continue

            if metric == 'chamfer':
                dist_mat = distance.cdist(
                    pred_lines[pred_id], gt_lines[i], 'euclidean')
                valid_ab = dist_mat.min(-1).mean()
                valid_ba = dist_mat.min(-2).mean()
                iou_matrix[pred_id, i] = -(valid_ba + valid_ab) / 2
            elif metric == 'iou':
                inter = o.intersection(pline).area
                union = o.union(pline).area
                iou_matrix[pred_id, i] = inter / union

    return iou_matrix


__all__ = ["custom_polyline_score"]
