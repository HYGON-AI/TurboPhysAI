# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/maptr/modules/encoder.py
# Upstream: projects/mmdet3d_plugin/maptr/modules/transformer.py
# Upstream: projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) 2022 Hust Vision Lab
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Reference compile boundaries and BEV layout for MapTRv2.

The optimized MapTRv2 checkout extracts eight helpers and decorates them with
``torch.compile``.  Those helper names do not exist in the official baseline,
so they cannot be recipe targets.  This module keeps the helpers local and
replaces their five existing callers instead.  The resulting call graph has the
same compile boundaries without editing model sources.
"""

from __future__ import annotations

import os

from .bev_pool_fix import normalize_bev_pool_layout


_COMPILED = {}
_ACTIVE_MODE = None


def _stack_metas(metas, key, device, dtype):
    import numpy as np
    import torch

    tensors = []
    for meta in metas:
        value = meta[key]
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif isinstance(value, list):
            value = torch.stack(
                [
                    torch.from_numpy(item) if isinstance(item, np.ndarray) else item
                    for item in value
                ],
                dim=0,
            )
        tensors.append(value)
    return torch.stack(tensors, dim=0).to(device=device, dtype=dtype)


def matmul_1(self, x, y, trans):
    del self
    import torch

    batch, cameras, _ = trans.shape
    points = torch.matmul(
        x.view(batch, cameras, 1, 1, 1, 3, 3), y.unsqueeze(-1)
    )
    return torch.cat(
        (
            points[..., :2, :] * points[..., 2:3, :],
            points[..., 2:3, :],
        ),
        dim=5,
    )


def matmul_2(self, x, y, trans, lidar2ego_trans, points):
    del self
    import torch

    batch, cameras, _ = trans.shape
    combine = torch.matmul(x, y)
    points = torch.matmul(
        combine.view(batch, cameras, 1, 1, 1, 3, 3), points
    ).squeeze(-1)
    points += trans.view(batch, cameras, 1, 1, 1, 3)
    points -= lidar2ego_trans.view(batch, 1, 1, 1, 1, 3)
    return points


def matmul_3(self, x, y, trans):
    del self
    import torch

    batch, cameras, _ = trans.shape
    return torch.matmul(
        x.view(batch, 1, 1, 1, 1, 3, 3), y.unsqueeze(-1)
    ).squeeze(-1)


def extract_metas(self, images, img_metas):
    device = images.device
    dtype = images.dtype
    lidar2img = _stack_metas(img_metas, "lidar2img", device, dtype)
    del lidar2img
    camera2ego = _stack_metas(img_metas, "camera2ego", device, dtype)
    camera_intrinsics = _stack_metas(
        img_metas, "camera_intrinsics", device, dtype
    )
    img_aug_matrix = _stack_metas(
        img_metas, "img_aug_matrix", device, dtype
    )
    lidar2ego = _stack_metas(img_metas, "lidar2ego", device, dtype)
    return (
        camera2ego[..., :3, :3],
        camera2ego[..., :3, 3],
        camera_intrinsics[..., :3, :3],
        img_aug_matrix[..., :3, :3],
        img_aug_matrix[..., :3, 3],
        lidar2ego[..., :3, :3],
        lidar2ego[..., :3, 3],
        camera2ego,
        camera_intrinsics,
    )


def down_sample(self, x):
    import torch

    x = x.permute(0, 4, 1, 2, 3).contiguous()
    x = torch.cat(x.unbind(dim=2), 1)
    x = x.permute(0, 1, 3, 2).contiguous()
    return self.downsample(x)


def initialize_queries_and_bev(
    self, object_query_embed, bev_embed, batch_size, bev_h, bev_w
):
    import torch

    query_pos, query = torch.split(
        object_query_embed, self.embed_dims, dim=1
    )
    query_pos = query_pos.unsqueeze(0).expand(batch_size, -1, -1)
    query = query.unsqueeze(0).expand(batch_size, -1, -1)
    reference_points = self.reference_points(query_pos).sigmoid()
    query = query.permute(1, 0, 2)
    query_pos = query_pos.permute(1, 0, 2)
    bev_embed = bev_embed.permute(1, 0, 2)
    spatial_shapes = torch.tensor([[bev_h, bev_w]], device=query.device)
    level_start_index = torch.tensor([0], device=query.device)
    return (
        query,
        bev_embed,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        reference_points,
    )


def compute_decoder_predictions(self, outputs, batch_size, num_vec, mlvl_feats):
    del mlvl_feats
    import torch
    from mmdet.models.utils.transformer import inverse_sigmoid

    bev_embed, depth, hidden_states, init_reference, inter_references = outputs
    hidden_states = hidden_states.permute(0, 2, 1, 3)
    groups = {
        "one2one": ([], [], []),
        "one2many": ([], [], []),
    }
    for level in range(hidden_states.shape[0]):
        if level == 0:
            reference = init_reference
        else:
            reference = inter_references[level - 1]
        reference = reference[..., : 3 if self.z_cfg["gt_z_flag"] else 2]
        reference = inverse_sigmoid(reference)
        output_class = self.cls_branches[level](
            hidden_states[level]
            .view(batch_size, num_vec, self.num_pts_per_vec, -1)
            .mean(2)
        )
        output_coord = self.reg_branches[level](hidden_states[level])
        output_coord = output_coord[
            ..., : 3 if self.z_cfg["gt_z_flag"] else 2
        ]
        output_coord = (output_coord + reference).sigmoid()
        output_coord, output_pts = self.transform_box(
            output_coord, num_vec=num_vec
        )
        split = self.num_vec_one2one
        groups["one2one"][0].append(output_class[:, :split])
        groups["one2one"][1].append(output_coord[:, :split])
        groups["one2one"][2].append(output_pts[:, :split])
        groups["one2many"][0].append(output_class[:, split:])
        groups["one2many"][1].append(output_coord[:, split:])
        groups["one2many"][2].append(output_pts[:, split:])

    return (
        bev_embed,
        *groups["one2one"],
        depth,
        *groups["one2many"],
    )


def compute_aux_seg_outputs(self, bev_embed, batch_size, mlvl_feats):
    """Run segmentation heads outside the compiled decoder prediction helper."""

    outputs_seg = None
    outputs_pv_seg = None
    if self.aux_seg["use_aux_seg"]:
        seg_bev_embed = (
            bev_embed.permute(1, 0, 2)
            .view(batch_size, self.bev_h, self.bev_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        if self.aux_seg["bev_seg"]:
            outputs_seg = self.seg_head(seg_bev_embed)
        if self.aux_seg["pv_seg"]:
            _, num_cam, _, feat_h, feat_w = mlvl_feats[-1].shape
            outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0, 1))
            outputs_pv_seg = outputs_pv_seg.view(
                batch_size, num_cam, -1, feat_h, feat_w
            )
    return outputs_seg, outputs_pv_seg


def prepare_transformer_inputs(self, mlvl_feats):
    import torch

    num_vec = self.num_vec if self.training else self.num_vec_one2one
    batch_size, _, _, _, _ = mlvl_feats[0].shape
    dtype = mlvl_feats[0].dtype
    if self.query_embed_type == "all_pts":
        object_query_embeds = self.query_embedding.weight.to(dtype)
    elif self.query_embed_type == "instance_pts":
        pts_embeds = self.pts_embedding.weight.unsqueeze(0)
        instance_embeds = self.instance_embedding.weight[
            0:num_vec
        ].unsqueeze(1)
        object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(
            dtype
        )
    else:
        raise ValueError(f"unsupported query_embed_type: {self.query_embed_type}")
    if self.bev_embedding is not None:
        bev_queries = self.bev_embedding.weight.to(dtype)
        bev_mask = torch.zeros(
            (batch_size, self.bev_h, self.bev_w), device=bev_queries.device
        ).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
    else:
        bev_queries = None
        bev_pos = None
    self_attn_mask = torch.zeros(
        [num_vec, num_vec], device=mlvl_feats[0].device
    ).bool()
    self_attn_mask[self.num_vec_one2one :, : self.num_vec_one2one] = True
    self_attn_mask[: self.num_vec_one2one, self.num_vec_one2one :] = True
    return (
        num_vec,
        object_query_embeds,
        bev_queries,
        bev_pos,
        self_attn_mask,
        batch_size,
    )


_HELPERS = {
    "matmul_1": matmul_1,
    "matmul_2": matmul_2,
    "matmul_3": matmul_3,
    "extract_metas": extract_metas,
    "down_sample": down_sample,
    "initialize_queries_and_bev": initialize_queries_and_bev,
    "compute_decoder_predictions": compute_decoder_predictions,
    "prepare_transformer_inputs": prepare_transformer_inputs,
}


def _prepare_helpers(options):
    global _ACTIVE_MODE
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return
    import torch

    mode = options.get("mode", "max-autotune-no-cudagraphs")
    _ACTIVE_MODE = mode
    for name, function in _HELPERS.items():
        key = (name, mode)
        if key not in _COMPILED:
            _COMPILED[key] = torch.compile(function, mode=mode)


def _helper(name):
    return _COMPILED.get((name, _ACTIVE_MODE), _HELPERS[name])


def _force_fp32(function, apply_to=None):
    from mmcv.runner import force_fp32

    if apply_to is None:
        return force_fp32()(function)
    return force_fp32(apply_to=apply_to)(function)


def base_transform_get_geometry_wrapper(original, options):
    del original
    _prepare_helpers(options)
    return _force_fp32(base_transform_get_geometry_v1)


def base_transform_get_geometry_v1(
    self,
    fH,
    fW,
    rots,
    trans,
    intrins,
    post_rots,
    post_trans,
    lidar2ego_rots,
    lidar2ego_trans,
    img_metas,
    **kwargs,
):
    import torch

    batch, cameras, _ = trans.shape
    if self.frustum is None:
        self.frustum = self.create_frustum(fH, fW, img_metas).to(trans.device)
    points = self.frustum - post_trans.view(batch, cameras, 1, 1, 1, 3)
    points = _helper("matmul_1")(
        self, torch.inverse(post_rots), points, trans
    )
    points = _helper("matmul_2")(
        self,
        rots,
        torch.inverse(intrins),
        trans,
        lidar2ego_trans,
        points,
    )
    points = _helper("matmul_3")(
        self, torch.inverse(lidar2ego_rots), points, trans
    )
    if "extra_rots" in kwargs:
        points = torch.matmul(
            kwargs["extra_rots"]
            .view(batch, 1, 1, 1, 1, 3, 3)
            .repeat(1, cameras, 1, 1, 1, 1, 1),
            points.unsqueeze(-1),
        ).squeeze(-1)
    if "extra_trans" in kwargs:
        points += kwargs["extra_trans"].view(
            batch, 1, 1, 1, 1, 3
        ).repeat(1, cameras, 1, 1, 1, 1)
    return points


def _bev_pool_5d(self, x, geom_feats):
    patched_pool = getattr(type(self), "bev_pool", None)
    if getattr(patched_pool, "_turbo_physai_returns_5d", False):
        return self.bev_pool(geom_feats, x, return_5d=True)

    import torch
    from mmdet3d.ops import bev_pool

    batch, cameras, depth, height, width, channels = x.shape
    point_count = batch * cameras * depth * height * width
    values = x.reshape(point_count, channels)
    coords = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
    coords = coords.view(point_count, 3)
    batch_indices = torch.cat(
        [
            torch.full(
                [point_count // batch, 1],
                index,
                device=values.device,
                dtype=torch.long,
            )
            for index in range(batch)
        ]
    )
    coords = torch.cat((coords, batch_indices), dim=1)
    kept = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < self.nx[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < self.nx[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < self.nx[2])
    )
    output = bev_pool(
        values[kept],
        coords[kept],
        batch,
        self.nx[2],
        self.nx[0],
        self.nx[1],
    )
    return normalize_bev_pool_layout(
        output, batch, channels, self.nx, channels_last=True
    )


def base_transform_forward_wrapper(original, options):
    del original
    _prepare_helpers(options)
    return _force_fp32(base_transform_forward)


def base_transform_forward(self, images, img_metas):
    (
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        camera2ego,
        camera_intrinsics,
    ) = _helper("extract_metas")(self, images, img_metas)
    height, width = images.shape[-2:]
    geometry = self.get_geometry_v1(
        height,
        width,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        img_metas,
    )
    mlp_input = self.get_mlp_input(
        camera2ego, camera_intrinsics, post_rots, post_trans
    )
    features, depth = self.get_cam_feats(images, mlp_input)
    return _bev_pool_5d(self, features, geometry), depth


def ls_transform_forward_wrapper(original, options):
    del original
    _prepare_helpers(options)
    return ls_transform_forward


def ls_transform_forward(self, images, img_metas):
    from projects.mmdet3d_plugin.maptr.modules.encoder import BaseTransform

    features, depth = BaseTransform.forward(self, images, img_metas)
    return {
        "bev": _helper("down_sample")(self, features),
        "depth": depth,
    }


def transformer_forward_wrapper(original, options):
    del original
    _prepare_helpers(options)
    return transformer_forward


def transformer_forward(
    self,
    mlvl_feats,
    lidar_feat,
    bev_queries,
    object_query_embed,
    bev_h,
    bev_w,
    grid_length=(0.512, 0.512),
    bev_pos=None,
    reg_branches=None,
    cls_branches=None,
    prev_bev=None,
    **kwargs,
):
    output_dict = self.get_bev_features(
        mlvl_feats,
        lidar_feat,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=grid_length,
        bev_pos=bev_pos,
        prev_bev=prev_bev,
        **kwargs,
    )
    bev_embed = output_dict["bev"]
    depth = output_dict["depth"]
    batch_size = mlvl_feats[0].size(0)
    (
        query,
        bev_embed,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        initial_reference,
    ) = _helper("initialize_queries_and_bev")(
        self, object_query_embed, bev_embed, batch_size, bev_h, bev_w
    )
    feat_flatten, feat_spatial_shapes, feat_level_start_index = (
        self.format_feats(mlvl_feats)
    )
    inter_states, inter_references = self.decoder(
        query=query,
        key=None,
        value=bev_embed,
        query_pos=query_pos,
        reference_points=reference_points,
        reg_branches=reg_branches,
        cls_branches=cls_branches,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        mlvl_feats=mlvl_feats,
        feat_flatten=feat_flatten,
        feat_spatial_shapes=feat_spatial_shapes,
        feat_level_start_index=feat_level_start_index,
        **kwargs,
    )
    return bev_embed, depth, inter_states, initial_reference, inter_references


def head_forward_wrapper(original, options):
    del original
    _prepare_helpers(options)
    return _force_fp32(
        head_forward, apply_to=("mlvl_feats", "prev_bev")
    )


def head_forward(
    self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False
):
    import torch

    (
        num_vec,
        object_query_embeds,
        bev_queries,
        bev_pos,
        self_attn_mask,
        batch_size,
    ) = _helper("prepare_transformer_inputs")(self, mlvl_feats)
    if only_bev:
        return self.transformer.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            img_metas=img_metas,
            prev_bev=prev_bev,
        )["bev"]
    outputs = self.transformer(
        mlvl_feats,
        lidar_feat,
        bev_queries,
        object_query_embeds,
        self.bev_h,
        self.bev_w,
        grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
        bev_pos=bev_pos,
        reg_branches=self.reg_branches if self.with_box_refine else None,
        cls_branches=self.cls_branches if self.as_two_stage else None,
        img_metas=img_metas,
        prev_bev=prev_bev,
        self_attn_mask=self_attn_mask,
        num_vec=num_vec,
        num_pts_per_vec=self.num_pts_per_vec,
    )
    predictions = _helper("compute_decoder_predictions")(
        self, outputs, batch_size, num_vec, mlvl_feats
    )
    (
        bev_embed,
        classes_one2one,
        coords_one2one,
        pts_one2one,
        depth,
        classes_one2many,
        coords_one2many,
        pts_one2many,
    ) = predictions
    outputs_seg, outputs_pv_seg = compute_aux_seg_outputs(
        self, bev_embed, batch_size, mlvl_feats
    )
    classes_one2one = torch.stack(classes_one2one)
    coords_one2one = torch.stack(coords_one2one)
    pts_one2one = torch.stack(pts_one2one)
    classes_one2many = torch.stack(classes_one2many)
    coords_one2many = torch.stack(coords_one2many)
    pts_one2many = torch.stack(pts_one2many)
    return {
        "bev_embed": bev_embed,
        "all_cls_scores": classes_one2one,
        "all_bbox_preds": coords_one2one,
        "all_pts_preds": pts_one2one,
        "enc_cls_scores": None,
        "enc_bbox_preds": None,
        "enc_pts_preds": None,
        "depth": depth,
        "seg": outputs_seg,
        "pv_seg": outputs_pv_seg,
        "one2many_outs": {
            "all_cls_scores": classes_one2many,
            "all_bbox_preds": coords_one2many,
            "all_pts_preds": pts_one2many,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "seg": None,
            "pv_seg": None,
        },
    }


__all__ = [
    "base_transform_forward_wrapper",
    "base_transform_get_geometry_wrapper",
    "head_forward_wrapper",
    "ls_transform_forward_wrapper",
    "transformer_forward_wrapper",
]
