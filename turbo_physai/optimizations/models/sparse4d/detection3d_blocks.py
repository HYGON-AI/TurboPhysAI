# Derived from Sparse4D: projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

"""Detection3D Blocks Optimization"""


import torch

from projects.mmdet3d_plugin.core.box3d import (
    X,
    Z,
    W,
    H,
    SIN_YAW,
    COS_YAW,
    VX,
)

from .compile import compile_wrapper


def box_encoder_forward(self, box_3d: torch.Tensor):
    pos_feat = self.pos_fc(box_3d[..., X:Z+1])
    size_feat = self.size_fc(box_3d[..., W:H+1])
    yaw_feat = self.yaw_fc(box_3d[..., SIN_YAW:COS_YAW+1])
    if self.mode == "add":
        output = pos_feat + size_feat + yaw_feat
    elif self.mode == "cat":
        output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

    if self.vel_dims > 0:
        vel_feat = self.vel_fc(box_3d[..., VX : VX + self.vel_dims])
        if self.mode == "add":
            output = output + vel_feat
        elif self.mode == "cat":
            output = torch.cat([output, vel_feat], dim=-1)
    if self.output_fc is not None:
        output = self.output_fc(output)
    return output


def box_encoder_forward_wrapper(_original, options):
    return compile_wrapper(box_encoder_forward, options)


def box_refinement_forward(
    self,
    instance_feature: torch.Tensor,
    anchor: torch.Tensor,
    anchor_embed: torch.Tensor,
    time_interval: torch.Tensor = 1.0,
    return_cls=True,
):
    feature = instance_feature + anchor_embed
    output = self.layers(feature)

    box_slice = slice(X, H + 1)
    box = output[..., box_slice] + anchor[..., box_slice]
    yaw_slice = slice(SIN_YAW, COS_YAW + 1)
    yaw = output[..., yaw_slice]
    if len(self.refine_state) > H - X + 1:
        yaw = yaw + anchor[..., yaw_slice]
    if self.normalize_yaw:
        yaw = torch.nn.functional.normalize(yaw, dim=-1)

    velocity = output[..., VX:]
    if self.output_dim > VX:
        if not isinstance(time_interval, torch.Tensor):
            time_interval = instance_feature.new_tensor(time_interval)
        translation = torch.transpose(velocity, 0, -1)
        velocity = torch.transpose(translation / time_interval, 0, -1)
        velocity = (velocity + anchor[..., VX:]).to(output.dtype)

    output = torch.cat((box, yaw, velocity), dim=-1)

    if return_cls:
        assert self.with_cls_branch, "Without classification layers !!!"
        cls = self.cls_layers(instance_feature)
    else:
        cls = None
    if return_cls and self.with_quality_estimation:
        quality = self.quality_layers(feature)
    else:
        quality = None
    return output, cls, quality


def box_refinement_forward_wrapper(_original, options):
    return compile_wrapper(box_refinement_forward, options)


def point_generator_forward(
    self,
    anchor,
    instance_feature=None,
    T_cur2temp_list=None,
    cur_timestamp=None,
    temp_timestamps=None,
):
    bs, num_anchor = anchor.shape[:2]
    size = anchor[..., None, W:H+1].exp()
    key_points = self.fix_scale * size
    if self.num_learnable_pts > 0 and instance_feature is not None:
        learnable_scale = (
            self.learnable_fc(instance_feature)
            .reshape(bs, num_anchor, self.num_learnable_pts, 3)
            .sigmoid()
            - 0.5
        )
        key_points = torch.cat(
            [key_points, learnable_scale * size], dim=-2
        )

    rotation_mat = anchor.new_zeros([bs, num_anchor, 3, 3])

    rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
    rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
    rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
    rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
    rotation_mat[:, :, 2, 2] = 1

    key_points = torch.matmul(
        rotation_mat[:, :, None], key_points[..., None]
    ).squeeze(-1)
    key_points = key_points + anchor[..., None, X:Z+1]

    if (
        cur_timestamp is None
        or temp_timestamps is None
        or T_cur2temp_list is None
        or len(temp_timestamps) == 0
    ):
        return key_points

    temp_key_points_list = []
    velocity = anchor[..., VX:]
    for i, t_time in enumerate(temp_timestamps):
        time_interval = cur_timestamp - t_time
        translation = (
            velocity
            * time_interval.to(dtype=velocity.dtype)[:, None, None]
        )
        temp_key_points = key_points - translation[:, :, None]
        T_cur2temp = T_cur2temp_list[i].to(dtype=key_points.dtype)
        temp_key_points = (
            T_cur2temp[:, None, None, :3]
            @ torch.cat(
                [
                    temp_key_points,
                    torch.ones_like(temp_key_points[..., :1]),
                ],
                dim=-1,
            ).unsqueeze(-1)
        )
        temp_key_points = temp_key_points.squeeze(-1)
        temp_key_points_list.append(temp_key_points)
    return key_points, temp_key_points_list



@staticmethod
def anchor_projection(
    anchor,
    T_src2dst_list,
    src_timestamp=None,
    dst_timestamps=None,
    time_intervals=None,
):
    dst_anchors = []
    for i in range(len(T_src2dst_list)):
        vel = anchor[..., VX:]
        vel_dim = vel.shape[-1]
        T_src2dst = torch.unsqueeze(
            T_src2dst_list[i].to(dtype=anchor.dtype), dim=1
        )

        center = anchor[..., :3] # anchor[..., [X, Y, Z]]
        if time_intervals is not None:
            time_interval = time_intervals[i]
        elif src_timestamp is not None and dst_timestamps is not None:
            time_interval = (src_timestamp - dst_timestamps[i]).to(
                dtype=vel.dtype
            )
        else:
            time_interval = None
        if time_interval is not None:
            translation = vel.transpose(0, -1) * time_interval
            translation = translation.transpose(0, -1)
            center = center - translation
        center = (
            torch.matmul(
                T_src2dst[..., :3, :3], center[..., None]
            ).squeeze(dim=-1)
            + T_src2dst[..., :3, 3]
        )
        size = anchor[..., W : H + 1]
        yaw_vec = torch.stack(
            (
                anchor[..., COS_YAW],
                anchor[..., SIN_YAW],
            ),
            dim=-1,
        )
        yaw = torch.matmul(
            T_src2dst[..., :2, :2],
            yaw_vec[..., None],
        ).squeeze(-1)
        vel = torch.matmul(
            T_src2dst[..., :vel_dim, :vel_dim], vel[..., None]
        ).squeeze(-1)
        dst_anchor = torch.cat([center, size, yaw, vel], dim=-1)
        # TODO: Fix bug
        # index = [X, Y, Z, W, L, H, COS_YAW, SIN_YAW] + [VX, VY, VZ][:vel_dim]
        # index = torch.tensor(index, device=dst_anchor.device)
        # index = torch.argsort(index)
        # dst_anchor = dst_anchor.index_select(dim=-1, index=index)
        dst_anchors.append(dst_anchor)
    return dst_anchors
