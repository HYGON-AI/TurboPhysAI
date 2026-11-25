# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py
# Upstream: projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py
# Upstream: projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py
# Upstream: projects/mmdet3d_plugin/bevformer/modules/encoder.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) OpenMMLab. All rights reserved.
# Modified from the MapTR/BEVFormer implementation by Zhiqi Li and Hust Vision Lab.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""v5 A1/A2 replacements with the baseline shapes and precision contract.

Each function keeps the baseline method body except for the loss denominator
(item -> zero-dimensional tensor) or the Boolean BEV mask's redundant cleanup.
Model helpers are resolved from the original modules at call time so existing
PhysAI replacements remain visible. Importing the catalog does not load this
implementation module.
"""

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmcv.utils import TORCH_VERSION, digit_version



# MIT License
#
# Copyright (c) 2022 Hust Vision Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def maptrv2_head_loss_single(self,
                cls_scores,
                bbox_preds,
                pts_preds,
                gt_bboxes_list,
                gt_labels_list,
                gt_shifts_pts_list,
                gt_bboxes_ignore_list=None):
    """"Loss function for outputs from a single decoder layer of a single
    feature level.
    Args:
        cls_scores (Tensor): Box score logits from a single decoder layer
            for all images. Shape [bs, num_query, cls_out_channels].
        bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
            for all images, with normalized coordinate (cx, cy, w, h) and
            shape [bs, num_query, 4].
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
        gt_pts_list (list[Tensor]): Ground truth pts for each image
            with shape (num_gts, fixed_num, 2) in [x,y] format.
        gt_bboxes_ignore_list (list[Tensor], optional): Bounding
            boxes which can be ignored for each image. Default None.
    Returns:
        dict[str, Tensor]: A dictionary of loss components for outputs from
            a single decoder layer.
    """
    from projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head import (
        normalize_2d_bbox,
        normalize_2d_pts,
        normalize_3d_pts,
        denormalize_2d_bbox,
        denormalize_2d_pts,
        denormalize_3d_pts,
        reduce_mean,
    )

    num_imgs = cls_scores.size(0)
    cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
    # import pdb;pdb.set_trace()
    cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                       gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                       gt_bboxes_ignore_list)
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     pts_targets_list, pts_weights_list,
     num_total_pos, num_total_neg) = cls_reg_targets
    # import pdb;pdb.set_trace()
    labels = torch.cat(labels_list, 0)
    label_weights = torch.cat(label_weights_list, 0)
    bbox_targets = torch.cat(bbox_targets_list, 0)
    bbox_weights = torch.cat(bbox_weights_list, 0)
    pts_targets = torch.cat(pts_targets_list, 0)
    pts_weights = torch.cat(pts_weights_list, 0)

    # classification loss
    cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    # construct weighted avg_factor to match with the official DETR repo
    cls_avg_factor = num_total_pos * 1.0 + \
        num_total_neg * self.bg_cls_weight
    if self.sync_cls_avg_factor:
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))

    cls_avg_factor = max(cls_avg_factor, 1)
    loss_cls = self.loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    # Compute the average number of gt boxes accross all gpus, for
    # normalization purposes
    num_total_pos = loss_cls.new_tensor([num_total_pos])
    num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).reshape(())

    # import pdb;pdb.set_trace()
    # regression L1 loss
    bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
    normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
    # normalized_bbox_targets = bbox_targets
    isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
    bbox_weights = bbox_weights * self.code_weights

    loss_bbox = self.loss_bbox(
        bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                           :4], bbox_weights[isnotnan, :4],
        avg_factor=num_total_pos)

    # regression pts CD loss
    # pts_preds = pts_preds
    # import pdb;pdb.set_trace()

    # num_samples, num_order, num_pts, num_coords
    normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                          else normalize_3d_pts(pts_targets, self.pc_range)

    # num_samples, num_pts, num_coords
    pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
    if self.num_pts_per_vec != self.num_pts_per_gt_vec:
        pts_preds = pts_preds.permute(0,2,1)
        pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                align_corners=True)
        pts_preds = pts_preds.permute(0,2,1).contiguous()

    # import pdb;pdb.set_trace()
    loss_pts = self.loss_pts(
        pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                        :,:],
        pts_weights[isnotnan,:,:],
        avg_factor=num_total_pos)
    dir_weights = pts_weights[:, :-self.dir_interval,0]
    denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                            else denormalize_3d_pts(pts_preds, self.pc_range)
    denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
    pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
    # dir_weights = pts_weights[:, indice,:-1,0]
    # import pdb;pdb.set_trace()
    loss_dir = self.loss_dir(
        denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                      :,:],
        dir_weights[isnotnan,:],
        avg_factor=num_total_pos)

    bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
    # regression IoU loss, defaultly GIoU loss
    loss_iou = self.loss_iou(
        bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
        avg_factor=num_total_pos)

    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_iou = torch.nan_to_num(loss_iou)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
    return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir


def maptr_head_loss_single(self,
                cls_scores,
                bbox_preds,
                pts_preds,
                gt_bboxes_list,
                gt_labels_list,
                gt_shifts_pts_list,
                gt_bboxes_ignore_list=None):
    """"Loss function for outputs from a single decoder layer of a single
    feature level.
    Args:
        cls_scores (Tensor): Box score logits from a single decoder layer
            for all images. Shape [bs, num_query, cls_out_channels].
        bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
            for all images, with normalized coordinate (cx, cy, w, h) and
            shape [bs, num_query, 4].
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
        gt_pts_list (list[Tensor]): Ground truth pts for each image
            with shape (num_gts, fixed_num, 2) in [x,y] format.
        gt_bboxes_ignore_list (list[Tensor], optional): Bounding
            boxes which can be ignored for each image. Default None.
    Returns:
        dict[str, Tensor]: A dictionary of loss components for outputs from
            a single decoder layer.
    """
    from projects.mmdet3d_plugin.maptr.dense_heads.maptr_head import (
        normalize_2d_bbox,
        normalize_2d_pts,
        denormalize_2d_bbox,
        denormalize_2d_pts,
        reduce_mean,
    )

    num_imgs = cls_scores.size(0)
    cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
    # import pdb;pdb.set_trace()
    cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                       gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                       gt_bboxes_ignore_list)
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     pts_targets_list, pts_weights_list,
     num_total_pos, num_total_neg) = cls_reg_targets
    # import pdb;pdb.set_trace()
    labels = torch.cat(labels_list, 0)
    label_weights = torch.cat(label_weights_list, 0)
    bbox_targets = torch.cat(bbox_targets_list, 0)
    bbox_weights = torch.cat(bbox_weights_list, 0)
    pts_targets = torch.cat(pts_targets_list, 0)
    pts_weights = torch.cat(pts_weights_list, 0)

    # classification loss
    cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    # construct weighted avg_factor to match with the official DETR repo
    cls_avg_factor = num_total_pos * 1.0 + \
        num_total_neg * self.bg_cls_weight
    if self.sync_cls_avg_factor:
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))

    cls_avg_factor = max(cls_avg_factor, 1)
    loss_cls = self.loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    # Compute the average number of gt boxes accross all gpus, for
    # normalization purposes
    num_total_pos = loss_cls.new_tensor([num_total_pos])
    num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).reshape(())

    # import pdb;pdb.set_trace()
    # regression L1 loss
    bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
    normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
    # normalized_bbox_targets = bbox_targets
    isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
    bbox_weights = bbox_weights * self.code_weights

    loss_bbox = self.loss_bbox(
        bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                           :4], bbox_weights[isnotnan, :4],
        avg_factor=num_total_pos)

    # regression pts CD loss
    # pts_preds = pts_preds
    # import pdb;pdb.set_trace()

    # num_samples, num_order, num_pts, num_coords
    normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

    # num_samples, num_pts, num_coords
    pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
    if self.num_pts_per_vec != self.num_pts_per_gt_vec:
        pts_preds = pts_preds.permute(0,2,1)
        pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                align_corners=True)
        pts_preds = pts_preds.permute(0,2,1).contiguous()

    # import pdb;pdb.set_trace()
    loss_pts = self.loss_pts(
        pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                        :,:],
        pts_weights[isnotnan,:,:],
        avg_factor=num_total_pos)
    dir_weights = pts_weights[:, :-self.dir_interval,0]
    denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
    denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
    pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
    # dir_weights = pts_weights[:, indice,:-1,0]
    # import pdb;pdb.set_trace()
    loss_dir = self.loss_dir(
        denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                      :,:],
        dir_weights[isnotnan,:],
        avg_factor=num_total_pos)

    bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
    # regression IoU loss, defaultly GIoU loss
    loss_iou = self.loss_iou(
        bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
        avg_factor=num_total_pos)

    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_iou = torch.nan_to_num(loss_iou)
        loss_pts = torch.nan_to_num(loss_pts)
        loss_dir = torch.nan_to_num(loss_dir)
    return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir


def bevformer_head_loss_single(self,
                cls_scores,
                bbox_preds,
                gt_bboxes_list,
                gt_labels_list,
                gt_bboxes_ignore_list=None):
    """"Loss function for outputs from a single decoder layer of a single
    feature level.
    Args:
        cls_scores (Tensor): Box score logits from a single decoder layer
            for all images. Shape [bs, num_query, cls_out_channels].
        bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
            for all images, with normalized coordinate (cx, cy, w, h) and
            shape [bs, num_query, 4].
        gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
            with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
        gt_labels_list (list[Tensor]): Ground truth class indices for each
            image with shape (num_gts, ).
        gt_bboxes_ignore_list (list[Tensor], optional): Bounding
            boxes which can be ignored for each image. Default None.
    Returns:
        dict[str, Tensor]: A dictionary of loss components for outputs from
            a single decoder layer.
    """
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import (
        normalize_bbox,
        reduce_mean,
    )

    num_imgs = cls_scores.size(0)
    cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
    bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
    cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                       gt_bboxes_list, gt_labels_list,
                                       gt_bboxes_ignore_list)
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     num_total_pos, num_total_neg) = cls_reg_targets
    labels = torch.cat(labels_list, 0)
    label_weights = torch.cat(label_weights_list, 0)
    bbox_targets = torch.cat(bbox_targets_list, 0)
    bbox_weights = torch.cat(bbox_weights_list, 0)

    # classification loss
    cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
    # construct weighted avg_factor to match with the official DETR repo
    cls_avg_factor = num_total_pos * 1.0 + \
        num_total_neg * self.bg_cls_weight
    if self.sync_cls_avg_factor:
        cls_avg_factor = reduce_mean(
            cls_scores.new_tensor([cls_avg_factor]))

    cls_avg_factor = max(cls_avg_factor, 1)
    loss_cls = self.loss_cls(
        cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

    # Compute the average number of gt boxes accross all gpus, for
    # normalization purposes
    num_total_pos = loss_cls.new_tensor([num_total_pos])
    num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).reshape(())

    # regression L1 loss
    bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
    normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
    isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
    bbox_weights = bbox_weights * self.code_weights

    loss_bbox = self.loss_bbox(
        bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                           :10], bbox_weights[isnotnan, :10],
        avg_factor=num_total_pos)
    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
    return loss_cls, loss_bbox


@force_fp32(apply_to=('reference_points', 'img_metas'))
def bevformer_point_sampling(self, reference_points, pc_range,  img_metas):

    lidar2img = []
    for img_meta in img_metas:
        lidar2img.append(img_meta['lidar2img'])
    lidar2img = np.asarray(lidar2img)
    lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
    reference_points = reference_points.clone()

    reference_points[..., 0:1] = reference_points[..., 0:1] * \
        (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * \
        (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * \
        (pc_range[5] - pc_range[2]) + pc_range[2]

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)

    reference_points = reference_points.permute(1, 0, 2, 3)
    D, B, num_query = reference_points.size()[:3]
    num_cam = lidar2img.size(1)

    reference_points = reference_points.view(
        D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)

    lidar2img = lidar2img.view(
        1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        reference_points.to(torch.float32)).squeeze(-1)
    eps = 1e-5

    bev_mask = (reference_points_cam[..., 2:3] > eps)
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1]
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0]

    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))

    reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    return reference_points_cam, bev_mask


__all__ = ['maptrv2_head_loss_single', 'maptr_head_loss_single', 'bevformer_head_loss_single', 'bevformer_point_sampling']
