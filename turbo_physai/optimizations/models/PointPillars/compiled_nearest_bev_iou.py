"""Compiled LiDAR nearest-BEV IoU replacement."""

import math
import os
from functools import wraps

import torch
from torch import Tensor
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


def _nearest_bev_lidar(boxes: Tensor) -> Tensor:
    centers = boxes[..., 0:2]
    dims = boxes[..., 3:5]
    yaw = boxes[..., 6:7]
    normed_yaw = torch.abs(
        yaw - torch.floor(yaw / math.pi + 0.5) * math.pi
    )
    swapped_dims = torch.cat((dims[..., 1:2], dims[..., 0:1]), dim=-1)
    dims = torch.where(normed_yaw > math.pi / 4, swapped_dims, dims)
    half_dims = dims * 0.5
    return torch.cat(
        (centers - half_dims, centers + half_dims), dim=-1
    )


def _nearest_bev_iou_lidar(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    bev1 = _nearest_bev_lidar(boxes1)
    bev2 = _nearest_bev_lidar(boxes2)
    area1 = (
        (bev1[..., 2] - bev1[..., 0])
        * (bev1[..., 3] - bev1[..., 1])
    )
    area2 = (
        (bev2[..., 2] - bev2[..., 0])
        * (bev2[..., 3] - bev2[..., 1])
    )
    left_top = torch.maximum(
        bev1[..., :, None, 0:2], bev2[..., None, :, 0:2]
    )
    right_bottom = torch.minimum(
        bev1[..., :, None, 2:4], bev2[..., None, :, 2:4]
    )
    wh = (right_bottom - left_top).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]
    union = area1[..., :, None] + area2[..., None, :] - intersection
    return intersection / union.clamp_min(1e-6)


_compiled_nearest_bev_iou_lidar = None


def _as_lidar_tensor(boxes):
    if isinstance(boxes, LiDARInstance3DBoxes):
        return boxes.tensor
    if isinstance(boxes, Tensor):
        return boxes
    return None


def compiled_nearest_bev_iou_wrapper(original, options):
    """Compile only the supported HCU path and preserve other call behavior."""

    @wraps(original)
    def wrapped(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        global _compiled_nearest_bev_iou_lidar
        tensor1 = _as_lidar_tensor(bboxes1)
        tensor2 = _as_lidar_tensor(bboxes2)
        supported = (
            self.coordinate == 'lidar'
            and mode == 'iou'
            and not is_aligned
            and tensor1 is not None
            and tensor2 is not None
        )
        if not supported:
            return original(self, bboxes1, bboxes2, mode, is_aligned)

        if (tensor1.shape[-1] != tensor2.shape[-1]
                or tensor1.shape[-1] < 7):
            return original(self, bboxes1, bboxes2, mode, is_aligned)
        if tensor1.shape[-2] == 0 or tensor2.shape[-2] == 0:
            return original(self, bboxes1, bboxes2, mode, is_aligned)
        if (tensor1.device.type == "cuda" and tensor2.device.type == "cuda"
                and os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") != "1"):
            if _compiled_nearest_bev_iou_lidar is None:
                _compiled_nearest_bev_iou_lidar = torch.compile(
                    _nearest_bev_iou_lidar, backend="inductor", dynamic=True,
                    fullgraph=True)
            return _compiled_nearest_bev_iou_lidar(tensor1, tensor2)
        return _nearest_bev_iou_lidar(tensor1, tensor2)

    return wrapped
