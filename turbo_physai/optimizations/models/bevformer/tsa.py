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

"""TemporalSelfAttention split-linear replacement."""

import torch
import torch.nn.functional as F


def _split_linear(self, linear, prev_query, query):
    split = self.embed_dims
    return (
        F.linear(prev_query, linear.weight[:, :split], linear.bias)
        + F.linear(query, linear.weight[:, split:], None)
    )


def temporal_self_attention_forward(
    self,
    query,
    key=None,
    value=None,
    identity=None,
    query_pos=None,
    key_padding_mask=None,
    reference_points=None,
    spatial_shapes=None,
    level_start_index=None,
    flag="decoder",
    **kwargs,
):
    del key, flag, kwargs
    if value is None:
        assert self.batch_first
        bs, len_bev, channels = query.shape
        value = torch.stack([query, query], 1).reshape(
            bs * 2, len_bev, channels
        )
    if identity is None:
        identity = query
    if query_pos is not None:
        query = query + query_pos
    if not self.batch_first:
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)
    bs, num_query, embed_dims = query.shape
    _, num_value, _ = value.shape
    assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
    assert self.num_bev_queue == 2

    prev_query = value[:bs]
    value = self.value_proj(value)
    if key_padding_mask is not None:
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
    value = value.reshape(
        bs * self.num_bev_queue, num_value, self.num_heads, -1
    )
    sampling_offsets = _split_linear(
        self, self.sampling_offsets, prev_query, query
    ).view(
        bs,
        num_query,
        self.num_heads,
        self.num_bev_queue,
        self.num_levels,
        self.num_points,
        2,
    )
    attention_weights = _split_linear(
        self, self.attention_weights, prev_query, query
    ).view(
        bs,
        num_query,
        self.num_heads,
        self.num_bev_queue,
        self.num_levels * self.num_points,
    )
    attention_weights = attention_weights.softmax(-1).view(
        bs,
        num_query,
        self.num_heads,
        self.num_bev_queue,
        self.num_levels,
        self.num_points,
    )
    attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(
        bs * self.num_bev_queue,
        num_query,
        self.num_heads,
        self.num_levels,
        self.num_points,
    ).contiguous()
    sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6).reshape(
        bs * self.num_bev_queue,
        num_query,
        self.num_heads,
        self.num_levels,
        self.num_points,
        2,
    )
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
        )
        sampling_locations = reference_points[:, :, None, :, None, :] + (
            sampling_offsets
            / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2]
            + sampling_offsets
            / self.num_points
            * reference_points[:, :, None, :, None, 2:]
            * 0.5
        )
    else:
        raise ValueError("reference_points last dimension must be 2 or 4")

    if torch.cuda.is_available() and value.is_cuda:
        from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
    else:
        from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
        output = multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
    output = output.permute(1, 2, 0).view(
        num_query, embed_dims, bs, self.num_bev_queue
    ).mean(-1).permute(2, 0, 1)
    output = self.output_proj(output)
    if not self.batch_first:
        output = output.permute(1, 0, 2)
    return self.dropout(output) + identity
