# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""
仅为训练构建的 MoGe teacher 跳过未使用的 normal head。
"""
from __future__ import annotations

import copy
import functools
import types

import torch
import torch.nn.functional as F


def wrap_infer(original, _options=None):
    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, _turbophysai_depth_only=False, **kwargs):
        nonlocal compiled
        if not _turbophysai_depth_only:
            return original(self, *args, **kwargs)
        if compiled is None:
            compiled = torch.compile(depth_teacher_forward)
        proxy = copy.copy(self)
        proxy.forward = types.MethodType(compiled, proxy)
        return original(proxy, *args, **kwargs)

    wrapped._turbophysai_depth_only_supported = True
    return wrapped


def depth_teacher_forward(self, image, num_tokens):
    from moge.utils.geometry_torch import normalized_view_plane_uv

    batch_size, _, img_h, img_w = image.shape
    device, dtype = image.device, image.dtype
    aspect_ratio = img_w / img_h
    base_h, base_w = (num_tokens / aspect_ratio) ** 0.5, (num_tokens * aspect_ratio) ** 0.5
    if isinstance(base_h, torch.Tensor):
        base_h, base_w = base_h.round().long(), base_w.round().long()
    else:
        base_h, base_w = round(base_h), round(base_w)

    features, cls_token = self.encoder(image, base_h, base_w, return_class_token=True)
    features = [features, None, None, None, None]
    for level in range(5):
        uv = normalized_view_plane_uv(
            width=base_w * 2 ** level, height=base_h * 2 ** level,
            aspect_ratio=aspect_ratio, dtype=dtype, device=device,
        )
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        features[level] = uv if features[level] is None else torch.concat([features[level], uv], dim=1)
    features = self.neck(features)

    points = self.points_head(features)[-1] if hasattr(self, "points_head") else None
    mask = self.mask_head(features)[-1] if hasattr(self, "mask_head") else None
    metric_scale = self.scale_head(cls_token) if hasattr(self, "scale_head") else None
    points, mask = (
        F.interpolate(v, (img_h, img_w), mode="bilinear", align_corners=False, antialias=False)
        if v is not None else None for v in (points, mask)
    )
    if points is not None:
        points = self._remap_points(points.permute(0, 2, 3, 1))
    if mask is not None:
        mask = mask.squeeze(1).sigmoid()
    if metric_scale is not None:
        metric_scale = metric_scale.squeeze(1).exp()
    return {k: v for k, v in {
        "points": points, "mask": mask, "metric_scale": metric_scale,
    }.items() if v is not None}
