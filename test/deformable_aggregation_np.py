# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np

class DeformableAggregationFunction():
    @staticmethod
    def golden_deformable_aggregation(feature_maps, spatial_shape, scale_start_index,
                                      sample_location, weights):
        batch_size = feature_maps.shape[0]
        num_feat = feature_maps.shape[1]
        num_embeds = feature_maps.shape[2]
        num_cams = spatial_shape.shape[0]
        num_scale = spatial_shape.shape[1]
        num_anchors = sample_location.shape[1]
        num_pts = sample_location.shape[2]
        num_groups = weights.shape[5]

        weights = weights.flatten()
        feature_maps = feature_maps.flatten()
        
        out = np.zeros((batch_size, num_anchors, num_embeds)).astype(np.float32)

        num_kernels = batch_size * num_anchors * num_pts * num_cams * num_scale
        for idx in range(num_kernels):
            chanenl_offset = 0
            weights_offset = idx
            scale_index = idx % num_scale
            idx //= num_scale

            cam_index = idx % num_cams
            idx //= num_cams

            pts_index = idx % num_pts
            idx //= num_pts

            anchor_index = idx % num_anchors
            idx //= num_anchors

            batch_index = idx % batch_size
            idx //= batch_size

            loc_w = sample_location[batch_index, anchor_index, pts_index, cam_index, 0]
            loc_h = sample_location[batch_index, anchor_index, pts_index, cam_index, 1]

            if loc_w <= 0 or loc_w >= 1:
                continue
            if loc_h <= 0 or loc_h >= 1:
                continue

            scale_start_index_idx = scale_start_index[cam_index, scale_index]
            value_offset = (batch_index * num_feat + scale_start_index_idx) * num_embeds

            h = spatial_shape[cam_index, scale_index, 0]
            w = spatial_shape[cam_index, scale_index, 1]

            h_im = loc_h * h - 0.5
            w_im = loc_w * w - 0.5

            h_low = np.floor(h_im).astype(int)
            w_low = np.floor(w_im).astype(int)
            h_high = h_low + 1
            w_high = w_low + 1
            lh = h_im - h_low
            lw = w_im - w_low
            hh = 1 - lh
            hw = 1 - lw

            w_stride = num_embeds
            h_stride = w * w_stride

            h_low_ptr_offset = h_low * h_stride
            h_high_ptr_offset = h_low_ptr_offset + h_stride

            w_low_ptr_offset = w_low * w_stride
            w_high_ptr_offset = w_low_ptr_offset + w_stride
            for groups_idx in range(num_groups):

                weights_idx = weights_offset * num_groups + groups_idx % num_groups
                weight = weights[weights_idx]

                v1 = 0
                if h_low >= 0 and w_low >= 0:
                    ptr1 = value_offset + h_low_ptr_offset + w_low_ptr_offset + chanenl_offset
                    v1 = feature_maps[ptr1 : ptr1 + num_embeds // num_groups]

                v2 = 0
                if h_low >= 0 and w_high <= w - 1:
                    ptr2 = value_offset + h_low_ptr_offset + w_high_ptr_offset + chanenl_offset
                    v2 = feature_maps[ptr2 : ptr2 + num_embeds // num_groups]

                v3 = 0
                if h_high <= h - 1 and w_low >= 0:
                    ptr3 = value_offset + h_high_ptr_offset + w_low_ptr_offset + chanenl_offset
                    v3 = feature_maps[ptr3 : ptr3 + num_embeds // num_groups]

                v4 = 0
                if h_high <= h - 1 and w_high <= w - 1:
                    ptr4 = value_offset + h_high_ptr_offset + w_high_ptr_offset + chanenl_offset
                    v4 = feature_maps[ptr4 : ptr4 + num_embeds // num_groups]

                w1 = hh * hw
                w2 = hh * lw
                w3 = lh * hw
                w4 = lh * lw

                val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4) * weight

                out[batch_index, anchor_index, chanenl_offset : chanenl_offset + num_embeds // num_groups] += val

                chanenl_offset += num_embeds // num_groups

        return out

    @staticmethod
    def golden_deformable_aggregation_grad(
        feature_maps,
        spatial_shape,
        scale_start_index,
        sample_location,
        weights
    ):
        batch_size = feature_maps.shape[0]
        num_feat = feature_maps.shape[1]
        num_embeds = feature_maps.shape[2]
        num_cams = spatial_shape.shape[0]
        num_scale = spatial_shape.shape[1]
        num_anchors = sample_location.shape[1]
        num_pts = sample_location.shape[2]
        num_groups = weights.shape[5]

        out_cpu = np.zeros((batch_size, num_anchors, num_embeds)).astype(np.float32)
        grad_mc_ms_feat = np.zeros_like(feature_maps)
        grad_sampling_location = np.zeros_like(sample_location)
        grad_weights = np.zeros_like(weights)
        grad_output = np.ones_like(out_cpu)

        feature_maps = feature_maps.flatten()
        spatial_shape = spatial_shape.flatten()
        scale_start_index = scale_start_index.flatten()
        sample_location = sample_location.flatten()
        weights = weights.flatten()
        grad_mc_ms_feat = grad_mc_ms_feat.flatten()
        grad_sampling_location = grad_sampling_location.flatten()
        grad_weights = grad_weights.flatten()
        grad_output = grad_output.flatten()

        num_kernels = batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale
        for idx in range(num_kernels):

            weights_ptr = idx // (num_embeds // num_groups)
            channel_index = idx % num_embeds
            idx //= num_embeds

            scale_index = idx % num_scale
            idx //= num_scale

            cam_index = idx % num_cams
            idx //= num_cams

            pts_index = idx % num_pts
            idx //= num_pts

            anchor_index = idx % num_anchors
            idx //= num_anchors

            batch_index = idx % batch_size

            anchor_index = batch_index * num_anchors + anchor_index
            loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1

            loc_w = sample_location[loc_offset]
            if loc_w <= 0 or loc_w >= 1:
                continue
            loc_h = sample_location[loc_offset + 1]
            if loc_h <= 0 or loc_h >= 1:
                continue

            grad = grad_output[anchor_index * num_embeds + channel_index]

            cam_scale_index = cam_index * num_scale + scale_index
            value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index

            cam_scale_index = cam_scale_index << 1

            h = spatial_shape[cam_scale_index]
            w = spatial_shape[cam_scale_index + 1]

            h_im = loc_h * h - 0.5
            w_im = loc_w * w - 0.5

            weight = weights[weights_ptr]

            h_low = np.floor(h_im).astype(int)
            w_low = np.floor(w_im).astype(int)
            h_high = h_low + 1
            w_high = w_low + 1
            lh = h_im - h_low
            lw = w_im - w_low
            hh = 1 - lh
            hw = 1 - lw

            w_stride = num_embeds
            h_stride = w * w_stride

            h_low_ptr_offset = h_low * h_stride
            h_high_ptr_offset = h_low_ptr_offset + h_stride

            w_low_ptr_offset = w_low * w_stride
            w_high_ptr_offset = w_low_ptr_offset + w_stride

            w1 = hh * hw
            w2 = hh * lw
            w3 = lh * hw
            w4 = lh * lw

            top_grad_mc_ms_feat = grad * weight

            grad_h_weight = 0
            grad_w_weight = 0

            v1 = 0
            if h_low >= 0 and w_low >= 0:
                ptr1 = value_offset + h_low_ptr_offset + w_low_ptr_offset
                v1 = feature_maps[ptr1]
                grad_h_weight -= hw * v1
                grad_w_weight -= hh * v1
                grad_mc_ms_feat[ptr1] += w1 * top_grad_mc_ms_feat

            v2 = 0
            if h_low >= 0 and w_high <= w - 1:
                ptr2 = value_offset + h_low_ptr_offset + w_high_ptr_offset
                v2 = feature_maps[ptr2]
                grad_h_weight -= lw * v2
                grad_w_weight += hh * v2
                grad_mc_ms_feat[ptr2] += w2 * top_grad_mc_ms_feat

            v3 = 0
            if h_high <= h - 1 and w_low >= 0:
                ptr3 = value_offset + h_high_ptr_offset + w_low_ptr_offset
                v3 = feature_maps[ptr3]
                grad_h_weight += hw * v3
                grad_w_weight -= lh * v3
                grad_mc_ms_feat[ptr3] += w3 * top_grad_mc_ms_feat

            v4 = 0
            if h_high <= h - 1 and w_high <= w - 1:
                ptr4 = value_offset + h_high_ptr_offset + w_high_ptr_offset
                v4 = feature_maps[ptr4]
                grad_h_weight += lw * v4
                grad_w_weight += lh * v4
                grad_mc_ms_feat[ptr4] += w4 * top_grad_mc_ms_feat

            val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4)

            grad_weights[weights_ptr] += grad * val

            grad_sampling_location[loc_offset] += w * grad_w_weight * top_grad_mc_ms_feat
            grad_sampling_location[loc_offset + 1] += h * grad_h_weight * top_grad_mc_ms_feat

        return grad_mc_ms_feat, grad_sampling_location, grad_weights
