# Derived from Sparse4D: projects/mmdet3d_plugin/models/sparse4d.py;
# projects/mmdet3d_plugin/ops/__init__.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

"""Use channels-last image features and cache Sparse4D packing metadata."""

import functools
from inspect import signature

import torch
from mmcv.runner import auto_fp16


def channels_last_init_wrapper(original, options):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.img_backbone.to(memory_format=torch.channels_last)
        self._feature_map_metadata_cache = None

    return wrapped


def feature_maps_format_wrapper(original, options):
    @functools.wraps(original)
    def wrapped(feature_maps, inverse=False, *, cached_metadata=None):
        if cached_metadata is None or inverse:
            return original(feature_maps, inverse=inverse)

        # Match the upstream forward packing order on cache hits.
        bs, num_cams = feature_maps[0].shape[:2]
        flattened = [
            feat.reshape(bs, num_cams, feat.shape[2], -1) for feat in feature_maps
        ]
        col_feats = torch.cat(flattened, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
        spatial_shape, scale_start_index = cached_metadata
        return [col_feats, spatial_shape, scale_start_index]

    return wrapped


@auto_fp16(apply_to=("img",), out_fp32=True)
def extract_feat(self, img, return_depth=False, metas=None):
    bs = img.shape[0]
    if img.dim() == 5:  # multi-view
        num_cams = img.shape[1]
        img = img.flatten(end_dim=1)
    else:
        num_cams = 1
    if self.use_grid_mask:
        img = self.grid_mask(img)
    img = img.to(memory_format=torch.channels_last)
    if "metas" in signature(self.img_backbone.forward).parameters:
        feature_maps = self.img_backbone(img, num_cams, metas=metas)
    else:
        feature_maps = self.img_backbone(img)
    img_neck = getattr(self, "img_neck", None)
    if img_neck is not None:
        feature_maps = img_neck(feature_maps)
    feature_maps = [feat.reshape(bs, num_cams, *feat.shape[1:]) for feat in feature_maps]
    if return_depth and self.depth_branch is not None:
        depths = self.depth_branch(feature_maps, metas.get("focal"))
    else:
        depths = None
    if self.use_deformable_func:
        from projects.mmdet3d_plugin import ops

        cache_key = (
            feature_maps[0].device,
            num_cams,
            tuple(feat.shape[-2:] for feat in feature_maps),
        )
        cache = self._feature_map_metadata_cache
        if cache is not None and cache[0] == cache_key:
            feature_maps = ops.feature_maps_format(
                feature_maps, cached_metadata=cache[1]
            )
        else:
            feature_maps = ops.feature_maps_format(feature_maps)
            self._feature_map_metadata_cache = (
                cache_key,
                (feature_maps[1], feature_maps[2]),
            )
    if return_depth:
        return feature_maps, depths
    return feature_maps
