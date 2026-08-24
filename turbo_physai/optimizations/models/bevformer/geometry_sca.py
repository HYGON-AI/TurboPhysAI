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

"""Tensorized BEV geometry and Spatial Cross Attention replacements."""

import numpy as np
import torch
from torchvision.transforms.functional import rotate

def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor(
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(0, 1, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[None, :, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Keep image features as (bs, num_cam, H*W, embed_dims) so SCA can
        # flatten bs*num_cam without materializing a layout copy.

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

def _cached_reference_points(self, H, W, Z=8, dim='3d', bs=1,
                                 device='cuda', dtype=torch.float):
        device = torch.device(device)
        key = (int(H), int(W), float(Z), int(self.num_points_in_pillar), dim,
               int(bs), device.type, device.index, str(dtype))
        cache = getattr(self, "_turbo_physai_reference_points_cache", None)
        if cache is None:
            cache = {}
            self._turbo_physai_reference_points_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        ref = self.get_reference_points(
            H,
            W,
            Z,
            self.num_points_in_pillar,
            dim=dim,
            bs=bs,
            device=device,
            dtype=dtype)
        cache[key] = ref
        return ref

def _point_sampling_tensor(reference_points, lidar2img, pc_range, img_h, img_w):
        reference_points = reference_points.clone()

        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(
            D, B, 1, num_query, 3).to(torch.float32)

        lidar2img = lidar2img.view(
            1, B, num_cam, 1, 4, 4).to(torch.float32)
        x = reference_points[..., 0]
        y = reference_points[..., 1]
        z = reference_points[..., 2]
        cam_x = lidar2img[..., 0, 0] * x + lidar2img[..., 0, 1] * y + \
            lidar2img[..., 0, 2] * z + lidar2img[..., 0, 3]
        cam_y = lidar2img[..., 1, 0] * x + lidar2img[..., 1, 1] * y + \
            lidar2img[..., 1, 2] * z + lidar2img[..., 1, 3]
        cam_z = lidar2img[..., 2, 0] * x + lidar2img[..., 2, 1] * y + \
            lidar2img[..., 2, 2] * z + lidar2img[..., 2, 3]
        eps = 1e-5

        denom = torch.maximum(cam_z, torch.ones_like(cam_z) * eps)
        ref_x = (cam_x / denom / img_w).unsqueeze(-1)
        ref_y = (cam_y / denom / img_h).unsqueeze(-1)
        reference_points_cam = torch.cat(
            (ref_x, ref_y), -1)

        bev_mask = ((cam_z.unsqueeze(-1) > eps) & (ref_y > 0.0)
                    & (ref_y < 1.0) & (ref_x < 1.0) & (ref_x > 0.0))

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        return reference_points_cam, bev_mask

def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4)
        reference_points_cam, bev_mask = _point_sampling_tensor(
            reference_points, lidar2img, pc_range,
            img_metas[0]['img_shape'][0][0],
            img_metas[0]['img_shape'][0][1])

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

def _prefetched_sca_inputs(self, img_metas, device):
        if not img_metas:
            return None
        keys = ('_bev_reference_points_cam', '_bev_bev_mask',
                '_bev_indexes', '_bev_index_lengths')
        if any(not all(key in meta for key in keys) for meta in img_metas):
            return None

        reference_points_cam = torch.stack(
            [meta['_bev_reference_points_cam'] for meta in img_metas],
            dim=1).to(device, non_blocking=True)
        bev_mask = torch.stack(
            [meta['_bev_bev_mask'] for meta in img_metas],
            dim=1).to(device, non_blocking=True)
        indexes = torch.stack(
            [meta['_bev_indexes'] for meta in img_metas],
            dim=0).to(device, non_blocking=True)
        index_lengths = torch.stack(
            [meta['_bev_index_lengths'] for meta in img_metas],
            dim=0).to(device, non_blocking=True)
        return reference_points_cam, bev_mask, indexes, index_lengths

def encoder_forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """

        output = bev_query
        intermediate = []

        ref_3d = _cached_reference_points(self,
            bev_h, bev_w, self.pc_range[5]-self.pc_range[2], dim='3d',
            bs=bev_query.size(1), device=bev_query.device,
            dtype=bev_query.dtype)
        ref_2d = _cached_reference_points(self,
            bev_h, bev_w, dim='2d', bs=bev_query.size(1),
            device=bev_query.device, dtype=bev_query.dtype)

        prefetched = _prefetched_sca_inputs(self,
            kwargs['img_metas'], bev_query.device)
        if prefetched is None:
            reference_points_cam, bev_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas'])
            ####################################
            cam_query_mask = (bev_mask.sum(-1) > 0).permute(1, 0, 2)
            index_lengths = cam_query_mask.sum(-1, dtype=torch.long)
            max_index_len = int(index_lengths.max().item())
            # Bucket the SCA query length to reduce torch.compile graph variants.
            for bucket_len in (9728, 10240, 11264, 12288):
                if max_index_len <= bucket_len:
                    max_index_len = min(cam_query_mask.size(-1), bucket_len)
                    break
            sorted_query_idx = torch.argsort(
                cam_query_mask.to(torch.int64), dim=-1, descending=True)
            indexes = sorted_query_idx[..., :max_index_len].contiguous()
            ####################################
        else:
            reference_points_cam, bev_mask, indexes, index_lengths = prefetched

        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        if prev_bev is not None:
            prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = torch.stack(
                [prev_bev, bev_query], 1).reshape(bs*2, len_bev, -1)
            hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)
        else:
            hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                bs*2, len_bev, num_bev_level, 2)

        for lid, layer in enumerate(self.layers):
            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam,
                bev_mask=bev_mask,
                prev_bev=prev_bev,
                ####################
                indexes=indexes,
                index_lengths=index_lengths,
                ####################
                **kwargs)

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

def spatial_cross_attention_forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                indexes=None,
                index_lengths=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        D = reference_points_cam.size(3)
        max_len = indexes.size(-1)
        valid_mask = torch.arange(
            max_len, device=query.device).view(1, 1, max_len) < index_lengths.unsqueeze(-1)

        # Each camera only attends to its visible BEV queries.
        gather_index = indexes.unsqueeze(-1).expand(
            bs, self.num_cams, max_len, self.embed_dims)
        queries_rebatch = torch.gather(
            query.unsqueeze(1).expand(bs, self.num_cams, num_query, self.embed_dims),
            2,
            gather_index)
        queries_rebatch = queries_rebatch * valid_mask.unsqueeze(-1)

        gather_ref_index = indexes.unsqueeze(-1).unsqueeze(-1).expand(
            bs, self.num_cams, max_len, D, 2)
        reference_points_rebatch = torch.gather(
            reference_points_cam.permute(1, 0, 2, 3, 4),
            2,
            gather_ref_index)
        reference_points_rebatch = reference_points_rebatch * valid_mask.unsqueeze(-1).unsqueeze(-1)

        bs, num_cams, l, embed_dims = value.shape
        value = value.reshape(bs * num_cams, l, embed_dims)
        key = value

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        queries = queries * valid_mask.unsqueeze(-1)
        scatter_index = indexes.reshape(bs, -1, 1).expand(bs, -1, self.embed_dims)
        slots.scatter_add_(
            1,
            scatter_index,
            queries.reshape(bs, -1, self.embed_dims))

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual
