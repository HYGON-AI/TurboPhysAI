# Copyright 2018-2019 OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""Complete TransFusionHead behavior and assignment transfer optimizations."""


def draw_heatmap_gaussian_cpu(
    heatmap, center, radius, k=1, gaussian_cache=None
):
    """Draw a cached Gaussian on a CPU heatmap without device synchronizations."""

    import torch

    cache_key = (radius, heatmap.dtype)
    gaussian = gaussian_cache.get(cache_key) if gaussian_cache is not None else None
    if gaussian is None:
        diameter = 2 * radius + 1
        sigma = diameter / 6
        coords = torch.arange(diameter, dtype=heatmap.dtype) - radius
        gaussian = torch.exp(
            -(coords[:, None].square() + coords[None, :].square())
            / (2 * sigma * sigma)
        )
        gaussian = torch.where(
            gaussian < torch.finfo(gaussian.dtype).eps * gaussian.max(),
            0,
            gaussian,
        )
        if gaussian_cache is not None:
            gaussian_cache[cache_key] = gaussian

    x, y = center
    height, width = heatmap.shape
    x0, x1 = max(x - radius, 0), min(x + radius + 1, width)
    y0, y1 = max(y - radius, 0), min(y + radius + 1, height)
    if x0 >= x1 or y0 >= y1:
        return heatmap
    gaussian_x0 = x0 - (x - radius)
    gaussian_y0 = y0 - (y - radius)
    gaussian_patch = gaussian[
        gaussian_y0 : gaussian_y0 + (y1 - y0),
        gaussian_x0 : gaussian_x0 + (x1 - x0),
    ]
    heatmap_patch = heatmap[y0:y1, x0:x1]
    torch.maximum(heatmap_patch, gaussian_patch * k, out=heatmap_patch)
    return heatmap


def gaussian_radius_cpu(det_size, min_overlap=0.5):
    """Uncompiled CPU equivalent of the TransFusion Gaussian radius helper."""

    import torch

    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + torch.sqrt(b1**2 - 4 * a1 * c1)) / 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + torch.sqrt(b2**2 - 4 * a2 * c2)) / 2
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + torch.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return torch.stack((r1, r2, r3)).min(0).values


def transfusion_init_wrapper(original, options):
    """Keep BEV positions as a movable, non-persistent module buffer."""

    del options
    import functools

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if "bev_pos" not in self._buffers:
            bev_pos = self.bev_pos
            del self.bev_pos
            self.register_buffer("bev_pos", bev_pos, persistent=False)

    return wrapped


def transfusion_forward_single(self, inputs, img_inputs, metas):
    """Optimized TransFusionHead forward path from BEVFusion."""

    del img_inputs, metas
    import torch
    from torch.nn import functional as functional

    batch_size = inputs.shape[0]
    lidar_feat = self.shared_conv(inputs)
    lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)
    bev_pos = self.bev_pos.repeat(batch_size, 1, 1)

    dense_heatmap = self.heatmap_head(lidar_feat)
    heatmap = dense_heatmap.detach().sigmoid()
    padding = self.nms_kernel_size // 2
    local_max = torch.zeros_like(heatmap)
    local_max_inner = functional.max_pool2d(
        heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0
    )
    local_max[:, :, padding:-padding, padding:-padding] = local_max_inner
    if self.test_cfg["dataset"] == "nuScenes":
        local_max[:, 8] = functional.max_pool2d(
            heatmap[:, 8], kernel_size=1, stride=1, padding=0
        )
        local_max[:, 9] = functional.max_pool2d(
            heatmap[:, 9], kernel_size=1, stride=1, padding=0
        )
    elif self.test_cfg["dataset"] == "Waymo":
        local_max[:, 1] = functional.max_pool2d(
            heatmap[:, 1], kernel_size=1, stride=1, padding=0
        )
        local_max[:, 2] = functional.max_pool2d(
            heatmap[:, 2], kernel_size=1, stride=1, padding=0
        )
    heatmap = heatmap * (heatmap == local_max)
    heatmap = heatmap.reshape(batch_size, heatmap.shape[1], -1)

    top_proposals = heatmap.reshape(batch_size, -1).argsort(
        dim=-1, descending=True
    )[..., : self.num_proposals]
    top_proposals_class = top_proposals // heatmap.shape[-1]
    top_proposals_index = top_proposals % heatmap.shape[-1]
    query_feat = lidar_feat_flatten.gather(
        index=top_proposals_index[:, None, :].expand(
            -1, lidar_feat_flatten.shape[1], -1
        ),
        dim=-1,
    )
    self.query_labels = top_proposals_class
    one_hot = functional.one_hot(
        top_proposals_class, num_classes=self.num_classes
    ).permute(0, 2, 1)
    query_feat += self.class_encoding(one_hot.float())
    query_pos = bev_pos.gather(
        index=top_proposals_index[:, None, :]
        .permute(0, 2, 1)
        .expand(-1, -1, bev_pos.shape[-1]),
        dim=1,
    )

    results = []
    for layer in range(self.num_decoder_layers):
        query_feat = self.decoder[layer](
            query_feat, lidar_feat_flatten, query_pos, bev_pos
        )
        layer_result = self.prediction_heads[layer](query_feat)
        layer_result["center"] = layer_result["center"] + query_pos.permute(0, 2, 1)
        results.append(layer_result)
        query_pos = layer_result["center"].detach().clone().permute(0, 2, 1)

    results[0]["query_heatmap_score"] = heatmap.gather(
        index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1),
        dim=-1,
    )
    results[0]["dense_heatmap"] = dense_heatmap
    if self.auxiliary is False:
        return [results[-1]]

    merged = {}
    for key in results[0].keys():
        if key not in ("dense_heatmap", "dense_heatmap_old", "query_heatmap_score"):
            merged[key] = torch.cat([result[key] for result in results], dim=-1)
        else:
            merged[key] = results[0][key]
    return [merged]


def transfusion_get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
    """Generate batch targets while transferring GT tensors asynchronously."""

    import numpy as np
    import torch
    from mmdet.core import multi_apply

    prediction_list = []
    gt_tensor_list = []
    for batch_index in range(len(gt_bboxes_3d)):
        prediction_list.append(
            {
                key: value[batch_index : batch_index + 1]
                for key, value in preds_dict[0].items()
            }
        )
        gt_tensor_list.append(
            gt_bboxes_3d[batch_index].tensor.to(
                gt_labels_3d[0].device, non_blocking=True
            )
        )
    results = multi_apply(
        self.get_targets_single,
        gt_bboxes_3d,
        gt_labels_3d,
        prediction_list,
        np.arange(len(gt_labels_3d)),
        gt_tensor_list,
    )
    return (
        torch.cat(results[0], dim=0),
        torch.cat(results[1], dim=0),
        torch.cat(results[2], dim=0),
        torch.cat(results[3], dim=0),
        torch.cat(results[4], dim=0),
        np.sum(results[5]),
        torch.stack(results[6]).mean(),
        torch.cat(results[7], dim=0),
    )


def transfusion_get_targets_single(
    self,
    gt_bboxes_3d,
    gt_labels_3d,
    preds_dict,
    batch_idx,
    gt_bboxes_tensor=None,
):
    """Generate one sample's assignment, regression and CPU heatmap targets."""

    import torch
    from mmdet.core.bbox.assigners import AssignResult

    if gt_bboxes_tensor is None:
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(
            gt_labels_3d.device, non_blocking=True
        )
    num_proposals = preds_dict["center"].shape[-1]
    score = preds_dict["heatmap"].detach()
    center = preds_dict["center"].detach()
    height = preds_dict["height"].detach()
    dimensions = preds_dict["dim"].detach()
    rotation = preds_dict["rot"].detach()
    velocity = preds_dict.get("vel")
    if velocity is not None:
        velocity = velocity.detach()
    boxes = self.bbox_coder.decode(
        score, rotation, dimensions, center, height, velocity
    )[0]["bboxes"]
    layer_count = self.num_decoder_layers if self.auxiliary else 1

    assignments = []
    for layer in range(layer_count):
        start = self.num_proposals * layer
        end = self.num_proposals * (layer + 1)
        layer_boxes = boxes[start:end, :]
        layer_score = score[..., start:end]
        if self.train_cfg.assigner.type == "HungarianAssigner3D":
            assignment = self.bbox_assigner.assign(
                layer_boxes,
                gt_bboxes_tensor,
                gt_labels_3d,
                layer_score,
                self.train_cfg,
            )
        elif self.train_cfg.assigner.type == "HeuristicAssigner":
            assignment = self.bbox_assigner.assign(
                layer_boxes,
                gt_bboxes_tensor,
                None,
                gt_labels_3d,
                self.query_labels[batch_idx],
            )
        else:
            raise NotImplementedError
        assignments.append(assignment)

    ensemble = AssignResult(
        num_gts=sum(result.num_gts for result in assignments),
        gt_inds=torch.cat([result.gt_inds for result in assignments]),
        max_overlaps=torch.cat([result.max_overlaps for result in assignments]),
        labels=torch.cat([result.labels for result in assignments]),
    )
    matched_rows = [
        result.get_extra_property("matched_row_inds") for result in assignments
    ]
    if all(rows is not None for rows in matched_rows):
        positive_indices = torch.cat(
            [
                rows + layer * self.num_proposals
                for layer, rows in enumerate(matched_rows)
            ]
        )
        assigned_gt_indices = torch.cat(
            [
                result.get_extra_property("matched_col_inds")
                for result in assignments
            ]
        )
        positive_gt_boxes = gt_bboxes_tensor[assigned_gt_indices]
    else:
        sampling = self.bbox_sampler.sample(ensemble, boxes, gt_bboxes_tensor)
        positive_indices = sampling.pos_inds
        assigned_gt_indices = sampling.pos_assigned_gt_inds
        positive_gt_boxes = sampling.pos_gt_bboxes

    bbox_targets = torch.zeros(
        [num_proposals, self.bbox_coder.code_size], device=center.device
    )
    bbox_weights = torch.zeros_like(bbox_targets)
    ious = torch.clamp(ensemble.max_overlaps, min=0.0, max=1.0)
    labels = boxes.new_zeros(num_proposals, dtype=torch.long)
    label_weights = boxes.new_ones(num_proposals, dtype=torch.long)
    if gt_labels_3d is not None:
        labels += self.num_classes
    if len(positive_indices) > 0:
        bbox_targets[positive_indices, :] = self.bbox_coder.encode(
            positive_gt_boxes
        )
        bbox_weights[positive_indices, :] = 1.0
        if gt_labels_3d is None:
            labels[positive_indices] = 1
        else:
            labels[positive_indices] = gt_labels_3d[assigned_gt_indices]
        if self.train_cfg.pos_weight > 0:
            label_weights[positive_indices] = self.train_cfg.pos_weight

    device = labels.device
    gt_bboxes_cpu = torch.cat(
        [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1
    )
    if gt_bboxes_cpu.device.type != "cpu":
        gt_bboxes_cpu = gt_bboxes_cpu.cpu()
    gt_labels_cpu = gt_labels_3d.detach().cpu()
    grid_size = self.train_cfg["grid_size"]
    output_factor = self.train_cfg["out_size_factor"]
    point_cloud_range = gt_bboxes_cpu.new_tensor(
        self.train_cfg["point_cloud_range"]
    )
    voxel_size = gt_bboxes_cpu.new_tensor(self.train_cfg["voxel_size"])
    feature_map_size = [
        grid_size[0] // output_factor,
        grid_size[1] // output_factor,
    ]
    heatmap_cpu = torch.zeros(
        (self.num_classes, feature_map_size[1], feature_map_size[0]),
        dtype=gt_bboxes_cpu.dtype,
        device="cpu",
        pin_memory=torch.cuda.is_available(),
    )
    if not hasattr(self, "_cpu_gaussian_cache"):
        self._cpu_gaussian_cache = {}
    widths = gt_bboxes_cpu[:, 3] / voxel_size[0] / output_factor
    lengths = gt_bboxes_cpu[:, 4] / voxel_size[1] / output_factor
    radii = gaussian_radius_cpu(
        (lengths, widths), min_overlap=self.train_cfg["gaussian_overlap"]
    )
    coordinate_x = (
        (gt_bboxes_cpu[:, 0] - point_cloud_range[0])
        / voxel_size[0]
        / output_factor
    ).to(torch.int32)
    coordinate_y = (
        (gt_bboxes_cpu[:, 1] - point_cloud_range[1])
        / voxel_size[1]
        / output_factor
    ).to(torch.int32)
    for index in range(len(gt_bboxes_cpu)):
        if widths[index] > 0 and lengths[index] > 0:
            radius = max(self.train_cfg["min_radius"], int(radii[index]))
            draw_heatmap_gaussian_cpu(
                heatmap_cpu[int(gt_labels_cpu[index])],
                (int(coordinate_y[index]), int(coordinate_x[index])),
                radius,
                gaussian_cache=self._cpu_gaussian_cache,
            )
    heatmap = heatmap_cpu.to(device, non_blocking=True)
    mean_iou = ious[positive_indices].sum() / max(len(positive_indices), 1)
    return (
        labels[None],
        label_weights[None],
        bbox_targets[None],
        bbox_weights[None],
        ious[None],
        int(positive_indices.shape[0]),
        mean_iou,
        heatmap[None],
    )


def transfusion_loss_wrapper(original, options):
    """Install the optimized loss while retaining MMCV's FP32 boundary."""

    del original, options
    from mmcv.runner import force_fp32

    return force_fp32(apply_to=("preds_dicts",))(transfusion_loss)


def _clip_sigmoid(value, eps=1e-4):
    # Match the reference implementation: sigmoid is in-place on the logits,
    # while clamp must produce a new tensor. A second in-place mutation bumps
    # the SigmoidBackward output version and breaks autograd.
    import torch

    return torch.clamp(value.sigmoid_(), min=eps, max=1 - eps)


def transfusion_loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
    """Optimized TransFusionHead loss without scalar GPU synchronization."""

    del kwargs
    import torch

    (
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        _ious,
        num_pos,
        matched_ious,
        heatmap,
    ) = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
    if hasattr(self, "on_the_image_mask"):
        label_weights = label_weights * self.on_the_image_mask
        bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
        num_pos = bbox_weights.max(-1).values.sum()
    predictions = preds_dicts[0][0]
    losses = {
        "loss_heatmap": self.loss_heatmap(
            _clip_sigmoid(predictions["dense_heatmap"]),
            heatmap,
            avg_factor=heatmap.eq(1).float().sum().clamp_min(1),
        )
    }

    layer_count = self.num_decoder_layers if self.auxiliary else 1
    for layer in range(layer_count):
        prefix = (
            "layer_-1"
            if layer == self.num_decoder_layers - 1
            or (layer == 0 and self.auxiliary is False)
            else f"layer_{layer}"
        )
        start = layer * self.num_proposals
        end = (layer + 1) * self.num_proposals
        layer_labels = labels[..., start:end].reshape(-1)
        layer_label_weights = label_weights[..., start:end].reshape(-1)
        layer_score = predictions["heatmap"][..., start:end]
        layer_class_score = layer_score.permute(0, 2, 1).reshape(
            -1, self.num_classes
        )
        layer_loss_cls = self.loss_cls(
            layer_class_score,
            layer_labels,
            layer_label_weights,
            avg_factor=max(num_pos, 1),
        )
        layer_center = predictions["center"][..., start:end]
        layer_height = predictions["height"][..., start:end]
        layer_rotation = predictions["rot"][..., start:end]
        layer_dimensions = predictions["dim"][..., start:end]
        parts = [layer_center, layer_height, layer_dimensions, layer_rotation]
        if "vel" in predictions:
            parts.append(predictions["vel"][..., start:end])
        encoded = torch.cat(parts, dim=1).permute(0, 2, 1)
        layer_bbox_weights = bbox_weights[:, start:end, :]
        regression_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(
            self.train_cfg.get("code_weights", None)
        )
        layer_loss_bbox = self.loss_bbox(
            encoded,
            bbox_targets[:, start:end, :],
            regression_weights,
            avg_factor=max(num_pos, 1),
        )
        losses[f"{prefix}_loss_cls"] = layer_loss_cls
        losses[f"{prefix}_loss_bbox"] = layer_loss_bbox
    losses["matched_ious"] = matched_ious.to(layer_loss_cls)
    return losses


def hungarian_assign(self, bboxes, gt_bboxes, gt_labels, cls_pred, train_cfg):
    """Pack Hungarian row/column indices into one host-to-device copy."""

    import numpy as np
    import torch
    from mmdet.core.bbox.assigners import AssignResult

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError('Please run "pip install scipy" to install scipy first.') from exc

    num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
    assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
    assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
    if num_gts == 0 or num_bboxes == 0:
        if num_gts == 0:
            assigned_gt_inds[:] = 0
        return AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)

    cls_cost = self.cls_cost(cls_pred[0].T, gt_labels)
    reg_cost = self.reg_cost(bboxes, gt_bboxes, train_cfg)
    iou = self.iou_calculator(bboxes, gt_bboxes)
    cost = (cls_cost + reg_cost + self.iou_cost(iou)).detach().cpu()
    matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
    matched_indices = torch.from_numpy(
        np.stack((matched_row_inds, matched_col_inds))
    ).to(bboxes.device)
    matched_row_inds, matched_col_inds = matched_indices.unbind(0)

    assigned_gt_inds[:] = 0
    assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
    assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
    max_overlaps = torch.zeros_like(iou.max(1).values)
    max_overlaps[matched_row_inds] = iou[matched_row_inds, matched_col_inds]
    result = AssignResult(
        num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels
    )
    result.set_extra_property("matched_row_inds", matched_row_inds)
    result.set_extra_property("matched_col_inds", matched_col_inds)
    return result
