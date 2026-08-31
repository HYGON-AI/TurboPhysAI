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

"""Factorized DepthLSSTransform feature materialization."""

import os
from typing import NamedTuple


class DepthFeatureFactorization(NamedTuple):
    """Depth probabilities (BN,D,H,W) and image features (BN,H,W,C)."""

    depth: object
    features: object


class PreparedGeometry(NamedTuple):
    """Corrected fused BEV coordinates ready for feature selection."""

    coords: object
    ranks: object
    kept: object
    batch_size: int


_DISABLED_SOFTMAX = None


def _resolve_bev_pool():
    """Use a patched public alias when present, otherwise the local frontend."""
    import inspect
    from mmdet3d.ops import bev_pool as public_bev_pool

    try:
        if "ranks" in inspect.signature(public_bev_pool).parameters:
            return public_bev_pool
    except (TypeError, ValueError):
        pass
    from turbo_physai.optimizations.common.mmdet3d.bev_pool import bev_pool

    return bev_pool


def use_bev_pool_prepare_opt():
    """Return whether native BEV-pool index preparation is enabled."""

    if os.environ.get("MMDET3D_DISABLE_BEV_POOL_PREPARE_OPT", "0") == "1":
        return False
    mode = os.environ.get("MMDET3D_BEV_POOL_PREPARE_OPT_MODE", "on").lower()
    return mode not in {"0", "false", "off", "disable", "disabled"}


def use_bev_pool_geometry_opt():
    """Return whether fused camera-to-BEV geometry is enabled."""

    if os.environ.get("MMDET3D_DISABLE_BEV_POOL_GEOMETRY_OPT", "0") == "1":
        return False
    mode = os.environ.get("MMDET3D_BEV_POOL_GEOMETRY_OPT_MODE", "on").lower()
    return mode in {"1", "true", "on", "enable", "enabled"}


def bev_pool_geometry_boundary_eps():
    return float(
        os.environ.get("MMDET3D_BEV_POOL_GEOMETRY_BOUNDARY_EPS", "1e-3")
    )


def bev_pool_geometry_correction_chunk():
    return int(
        os.environ.get(
            "MMDET3D_BEV_POOL_GEOMETRY_CORRECTION_CHUNK", "262144"
        )
    )


def base_transform_init_wrapper(original, options):
    """Install BaseTransform constants and methods added by optimized base.py."""

    del options
    import functools

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        # The optimized DepthLSSTransform constructor explicitly passes
        # add_depth_features=False. Existing subclasses were registered before
        # patching, so reproduce that constructor behavior after the baseline
        # BaseTransform initializer has run.
        if type(self).__name__ == "DepthLSSTransform":
            self.add_depth_features = False
        self._bev_output_shape = tuple(int(value) for value in self.nx)

        # Existing registered subclasses keep the original BaseTransform class
        # object. Attach methods introduced by the optimized class to that base
        # object so every already-defined subclass sees the complete API.
        for cls in type(self).__mro__:
            if cls.__name__ == "BaseTransform":
                cls.bev_pool_prepared = base_transform_bev_pool_prepared
                cls.bev_pool_prepared_factorized = (
                    base_transform_bev_pool_prepared_factorized
                )
                cls.correct_bev_pool_geometry_boundaries = (
                    base_transform_correct_bev_pool_geometry_boundaries
                )
                break

    return wrapped


def _softmax_impl(x):
    return x.softmax(dim=1)


def materialized_softmax(x):
    """Materialize depth probabilities outside the surrounding compiled graph."""

    global _DISABLED_SOFTMAX
    if _DISABLED_SOFTMAX is None:
        import torch

        _DISABLED_SOFTMAX = torch.compiler.disable(_softmax_impl)
    return _DISABLED_SOFTMAX(x)


def depth_lss_get_cam_feats_wrapper(original, options):
    """Prepare the softmax graph break before installing the optimized method."""

    del original, options
    global _DISABLED_SOFTMAX
    import torch

    _DISABLED_SOFTMAX = torch.compiler.disable(_softmax_impl)
    return depth_lss_get_cam_feats


def gather_factorized_depth_features(factors, kept_indices):
    """Materialize only rows that survive BEV geometry filtering."""

    import torch

    depth, features = factors
    batch_cameras, depth_bins, height, width = depth.shape
    channels = features.shape[-1]
    points_per_camera = depth_bins * height * width
    pixels_per_camera = height * width
    selected_depth = torch.index_select(depth.reshape(-1), 0, kept_indices)
    feature_indices = (
        torch.div(kept_indices, points_per_camera, rounding_mode="floor")
        * pixels_per_camera
        + kept_indices.remainder(pixels_per_camera)
    )
    selected_features = torch.index_select(
        features.reshape(batch_cameras * pixels_per_camera, channels),
        0,
        feature_indices,
    )
    return selected_features * selected_depth.unsqueeze(1)


def depth_lss_get_cam_feats(self, x, d, mats_dict=None):
    """Return the depth/feature factors instead of their dense outer product."""

    del mats_dict
    import torch

    batch, cameras, channels, feature_h, feature_w = x.shape
    d = d.view(batch * cameras, *d.shape[2:])
    x = x.view(batch * cameras, channels, feature_h, feature_w)
    d = self.dtransform(d)
    x = torch.cat([d, x], dim=1)
    x = self.depthnet(x)

    # The softmax is materialized only at the factor boundary. The much larger
    # D x H x W x C outer product remains deferred until geometry filtering.
    depth = materialized_softmax(x[:, : self.D])
    image_features = x[:, self.D : (self.D + self.C)]
    image_features = image_features.permute(0, 2, 3, 1).contiguous()
    return DepthFeatureFactorization(depth, image_features)


def base_depth_transform_forward_wrapper(original, options):
    """Install the optimized BaseDepthTransform forward with MMCV FP32 casting."""

    del original, options
    from mmcv.runner import force_fp32

    return force_fp32()(base_depth_transform_forward)


def base_depth_transform_forward(
    self,
    img,
    points,
    radar,
    sensor2ego,
    lidar2ego,
    lidar2camera,
    lidar2image,
    cam_intrinsic,
    camera2lidar,
    img_aug_matrix,
    lidar_aug_matrix,
    metas,
    **kwargs,
):
    """Optimized BaseDepthTransform implementation from BEVFusion.

    Point-to-image writes are flattened and vectorized across cameras. Geometry
    and BEV pooling deliberately dispatch through ``self`` so the existing
    TurboPhysAI geometry, factorization, and native pooling patches compose with
    this complete class method.
    """

    del lidar2camera, metas, kwargs
    import torch

    intrins = cam_intrinsic[..., :3, :3]
    post_rots = img_aug_matrix[..., :3, :3]
    post_trans = img_aug_matrix[..., :3, 3]
    camera2lidar_rots = camera2lidar[..., :3, :3]
    camera2lidar_trans = camera2lidar[..., :3, 3]

    if self.use_points == "radar":
        points = radar

    height_values = torch.arange(
        0.25,
        2.25,
        0.25,
        device=points[0].device,
        dtype=points[0].dtype,
    )
    if self.height_expand:
        for batch_index in range(len(points)):
            point_count = points[batch_index].shape[0]
            repeated = points[batch_index].repeat_interleave(8, dim=0)
            repeated[:, 2] = height_values.repeat(point_count)
            points[batch_index] = repeated

    batch_size = len(points)
    depth_in_channels = 1 if self.depth_input == "scalar" else self.D
    if self.add_depth_features:
        depth_in_channels += points[0].shape[1]

    depth = torch.zeros(
        batch_size,
        img.shape[1],
        depth_in_channels,
        *self.image_size,
        device=points[0].device,
    )

    for batch_index in range(batch_size):
        current_coords = points[batch_index][:, :3]
        current_img_aug = img_aug_matrix[batch_index]
        current_lidar_aug = lidar_aug_matrix[batch_index]
        current_lidar2image = lidar2image[batch_index]

        current_coords -= current_lidar_aug[:3, 3]
        current_coords = torch.linalg.inv_ex(
            current_lidar_aug[:3, :3], check_errors=False
        )[0].matmul(current_coords.transpose(1, 0))
        current_coords = current_lidar2image[:, :3, :3].matmul(current_coords)
        current_coords += current_lidar2image[:, :3, 3].reshape(-1, 3, 1)

        distances = current_coords[:, 2, :]
        current_coords[:, 2, :] = torch.clamp(
            current_coords[:, 2, :], 1e-5, 1e5
        )
        current_coords[:, :2, :] /= current_coords[:, 2:3, :]
        current_coords = current_img_aug[:, :3, :3].matmul(current_coords)
        current_coords += current_img_aug[:, :3, 3].reshape(-1, 3, 1)
        current_coords = current_coords[:, :2, :].transpose(1, 2)
        current_coords = torch.flip(current_coords, dims=(-1,))

        on_image = (
            (current_coords[..., 0] < self.image_size[0])
            & (current_coords[..., 0] >= 0)
            & (current_coords[..., 1] < self.image_size[1])
            & (current_coords[..., 1] >= 0)
        )
        valid_pairs = torch.nonzero(on_image, as_tuple=False)
        camera_indices = valid_pairs[:, 0]
        point_indices = valid_pairs[:, 1]
        points_per_camera = on_image.shape[1]
        projection_indices = camera_indices * points_per_camera + point_indices
        masked_coords = torch.index_select(
            current_coords.reshape(-1, 2), 0, projection_indices
        ).long()
        masked_distances = torch.index_select(
            distances.reshape(-1), 0, projection_indices
        )

        image_area = self.image_size[0] * self.image_size[1]
        camera_stride = depth_in_channels * image_area
        pixel_indices = (
            masked_coords[:, 0] * self.image_size[1] + masked_coords[:, 1]
        )
        camera_offsets = camera_indices * camera_stride
        flat_depth = depth[batch_index].reshape(-1)

        if self.depth_input == "scalar":
            flat_depth.index_copy_(
                0, camera_offsets + pixel_indices, masked_distances
            )
        elif self.depth_input == "one-hot":
            depth_bins = torch.clamp(masked_distances, max=self.D - 1).long()
            depth_bins = torch.where(
                depth_bins < 0, depth_bins + self.D, depth_bins
            )
            flat_depth.index_fill_(
                0,
                camera_offsets + depth_bins * image_area + pixel_indices,
                1.0,
            )

        if self.add_depth_features:
            feature_count = points[batch_index].shape[-1]
            point_features = torch.index_select(
                points[batch_index], 0, point_indices
            )
            feature_channels = torch.arange(
                depth_in_channels - feature_count,
                depth_in_channels,
                device=depth.device,
                dtype=torch.long,
            )
            feature_indices = (
                camera_offsets.unsqueeze(1)
                + feature_channels.unsqueeze(0) * image_area
                + pixel_indices.unsqueeze(1)
            )
            flat_depth.index_copy_(
                0, feature_indices.reshape(-1), point_features.reshape(-1)
            )

    extra_rots = lidar_aug_matrix[..., :3, :3]
    extra_trans = lidar_aug_matrix[..., :3, 3]
    geometry = self.get_geometry(
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        extra_rots=extra_rots,
        extra_trans=extra_trans,
    )
    matrices = {
        "intrin_mats": intrins,
        "ida_mats": img_aug_matrix,
        "bda_mat": lidar_aug_matrix,
        "sensor2ego_mats": sensor2ego,
    }
    features = self.get_cam_feats(img, depth, matrices)

    returns_depth = False
    if type(features) is tuple:
        features, predicted_depth = features
        returns_depth = True

    output = self.bev_pool(geometry, features)
    if returns_depth:
        return output, predicted_depth
    return output


def _output_shape(model):
    return (
        int((model.xbound[1] - model.xbound[0]) / model.xbound[2]),
        int((model.ybound[1] - model.ybound[0]) / model.ybound[2]),
        int((model.zbound[1] - model.zbound[0]) / model.zbound[2]),
    )


def _correct_geometry_boundaries(
    self,
    coords,
    ranks,
    kept,
    boundary,
    inv_post_rots,
    post_trans,
    combine,
    camera2lidar_trans,
    extra_rots,
    extra_trans,
    batch_size,
):
    """Recompute only points close enough to a voxel boundary to be ambiguous."""

    import torch

    candidates = torch.nonzero(boundary, as_tuple=False).flatten()
    if candidates.numel() == 0:
        return coords, ranks, kept

    geom_depth, geom_height, geom_width = self.frustum.shape[:3]
    cameras = inv_post_rots.shape[1]
    _, output_width, output_depth = _output_shape(self)
    chunk_size = bev_pool_geometry_correction_chunk()
    if chunk_size <= 0:
        raise ValueError("BEV geometry correction chunk must be positive")
    base = self.bx - self.dx / 2.0

    for start in range(0, candidates.numel(), chunk_size):
        current = candidates[start : start + chunk_size]
        flat = current
        width_index = flat % geom_width
        flat = torch.div(flat, geom_width, rounding_mode="floor")
        height_index = flat % geom_height
        flat = torch.div(flat, geom_height, rounding_mode="floor")
        depth_index = flat % geom_depth
        flat = torch.div(flat, geom_depth, rounding_mode="floor")
        camera_index_in_batch = flat % cameras
        batch_index = torch.div(flat, cameras, rounding_mode="floor")

        frustum_index = (
            (depth_index * geom_height + height_index) * geom_width
            + width_index
        )
        camera_index = batch_index * cameras + camera_index_in_batch
        points = torch.index_select(
            self.frustum.reshape(-1, 3), 0, frustum_index
        )
        points -= torch.index_select(post_trans.reshape(-1, 3), 0, camera_index)
        points = torch.index_select(
            inv_post_rots.reshape(-1, 3, 3), 0, camera_index
        ).matmul(points.unsqueeze(-1))
        points = torch.cat((points[:, :2] * points[:, 2:3], points[:, 2:3]), 1)
        points = torch.index_select(
            combine.reshape(-1, 3, 3), 0, camera_index
        ).matmul(points).squeeze(-1)
        points += torch.index_select(
            camera2lidar_trans.reshape(-1, 3), 0, camera_index
        )
        points = torch.index_select(extra_rots, 0, batch_index).matmul(
            points.unsqueeze(-1)
        ).squeeze(-1)
        points += torch.index_select(extra_trans, 0, batch_index)

        current_coords = ((points - base) / self.dx).long()
        current_kept = (
            (current_coords[:, 0] >= 0)
            & (current_coords[:, 0] < self.nx[0])
            & (current_coords[:, 1] >= 0)
            & (current_coords[:, 1] < self.nx[1])
            & (current_coords[:, 2] >= 0)
            & (current_coords[:, 2] < self.nx[2])
        )
        current_ranks = (
            current_coords[:, 0] * (output_width * output_depth * batch_size)
            + current_coords[:, 1] * (output_depth * batch_size)
            + current_coords[:, 2] * batch_size
            + batch_index
        ).to(torch.int32)
        current_ranks = torch.where(
            current_kept, current_ranks, torch.full_like(current_ranks, -1)
        )
        corrected_coords = torch.cat(
            (
                current_coords.to(torch.int32),
                batch_index.to(torch.int32).unsqueeze(1),
            ),
            dim=1,
        )
        coords.index_copy_(0, current, corrected_coords)
        ranks.index_copy_(0, current, current_ranks)
        kept.index_copy_(0, current, current_kept)

    return coords, ranks, kept


def dense_base_transform_get_geometry(
    self,
    camera2lidar_rots,
    camera2lidar_trans,
    intrins,
    post_rots,
    post_trans,
    **kwargs,
):
    """Optimized dense BaseTransform geometry using non-synchronizing inverses."""

    import torch

    batch_size, cameras, _ = camera2lidar_trans.shape
    points = self.frustum - post_trans.view(
        batch_size, cameras, 1, 1, 1, 3
    )
    points = (
        torch.linalg.inv_ex(post_rots, check_errors=False)[0]
        .view(batch_size, cameras, 1, 1, 1, 3, 3)
        .matmul(points.unsqueeze(-1))
    )
    points = torch.cat(
        (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]),
        dim=5,
    )
    combine = camera2lidar_rots.matmul(
        torch.linalg.inv_ex(intrins, check_errors=False)[0]
    )
    points = combine.view(
        batch_size, cameras, 1, 1, 1, 3, 3
    ).matmul(points).squeeze(-1)
    points += camera2lidar_trans.view(batch_size, cameras, 1, 1, 1, 3)

    if "extra_rots" in kwargs:
        points = (
            kwargs["extra_rots"]
            .view(batch_size, 1, 1, 1, 1, 3, 3)
            .repeat(1, cameras, 1, 1, 1, 1, 1)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )
    if "extra_trans" in kwargs:
        points += kwargs["extra_trans"].view(
            batch_size, 1, 1, 1, 1, 3
        ).repeat(1, cameras, 1, 1, 1, 1)
    return points


def base_transform_get_geometry_wrapper(original, options):
    """Retain BaseTransform's MMCV FP32 boundary around optimized geometry."""

    del original, options
    from mmcv.runner import force_fp32

    return force_fp32()(base_transform_get_geometry)


def base_transform_get_geometry(
    self,
    camera2lidar_rots,
    camera2lidar_trans,
    intrins,
    post_rots,
    post_trans,
    **kwargs,
):
    """Use fused geometry when possible and optimized dense geometry otherwise."""

    import torch
    from turbo_physai.optimizations.common.mmdet3d.bev_pool import (
        bev_pool_prepare_geometry,
    )

    tensors = (camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans)
    if (
        not use_bev_pool_geometry_opt()
        or not camera2lidar_trans.is_cuda
        or any(tensor.dtype != torch.float32 for tensor in tensors)
    ):
        return dense_base_transform_get_geometry(
            self,
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            **kwargs,
        )

    batch_size, _, _ = camera2lidar_trans.shape
    extra_rots = kwargs.get("extra_rots")
    if extra_rots is None:
        extra_rots = torch.eye(
            3, device=camera2lidar_trans.device, dtype=camera2lidar_trans.dtype
        ).expand(batch_size, -1, -1).contiguous()
    extra_trans = kwargs.get("extra_trans")
    if extra_trans is None:
        extra_trans = torch.zeros(
            (batch_size, 3),
            device=camera2lidar_trans.device,
            dtype=camera2lidar_trans.dtype,
        )
    if extra_rots.dtype != torch.float32 or extra_trans.dtype != torch.float32:
        return dense_base_transform_get_geometry(
            self,
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            **kwargs,
        )

    inv_post_rots = torch.linalg.inv_ex(post_rots, check_errors=False)[0]
    combine = camera2lidar_rots.matmul(
        torch.linalg.inv_ex(intrins, check_errors=False)[0]
    )
    output_height, output_width, output_depth = _output_shape(self)
    boundary_eps = bev_pool_geometry_boundary_eps()
    if boundary_eps < 0:
        raise ValueError("BEV geometry boundary epsilon must be non-negative")
    coords, ranks, kept, boundary = bev_pool_prepare_geometry(
        self.frustum,
        inv_post_rots,
        post_trans,
        combine,
        camera2lidar_trans,
        extra_rots,
        extra_trans,
        self.bx,
        self.dx,
        self.nx,
        batch_size,
        output_depth,
        output_height,
        output_width,
        boundary_eps=boundary_eps,
    )
    coords, ranks, kept = _correct_geometry_boundaries(
        self,
        coords,
        ranks,
        kept,
        boundary,
        inv_post_rots,
        post_trans,
        combine,
        camera2lidar_trans,
        extra_rots,
        extra_trans,
        batch_size,
    )
    return PreparedGeometry(coords, ranks, kept, batch_size)


def base_transform_bev_pool_wrapper(original, options):
    """Retain BaseTransform's MMCV FP32 boundary around optimized pooling."""

    del original, options
    from mmcv.runner import force_fp32

    return force_fp32()(base_transform_bev_pool)


def base_transform_bev_pool_prepared(self, x, coords, ranks, kept):
    """Pool dense features using geometry prepared by the fused kernel."""

    import torch
    bev_pool = _resolve_bev_pool()

    batch, cameras, depth_bins, height, width, channels = x.shape
    output_height, output_width, output_depth = _output_shape(self)
    values = x.reshape(
        batch * cameras * depth_bins * height * width, channels
    )
    kept_indices = torch.nonzero(kept, as_tuple=False).flatten()
    values = torch.index_select(values, 0, kept_indices)
    coords = torch.index_select(coords, 0, kept_indices)
    ranks = torch.index_select(ranks, 0, kept_indices)
    values = bev_pool(
        values,
        coords,
        batch,
        output_depth,
        output_height,
        output_width,
        ranks=ranks,
    )
    return torch.cat(values.unbind(dim=2), 1)


def base_transform_bev_pool_prepared_factorized(
    self, factors, coords, ranks, kept, batch_size
):
    """Pool a factorized depth/image outer product after geometry filtering."""

    import torch
    bev_pool = _resolve_bev_pool()

    kept_indices = torch.nonzero(kept, as_tuple=False).flatten()
    values = gather_factorized_depth_features(factors, kept_indices)
    coords = torch.index_select(coords, 0, kept_indices)
    ranks = torch.index_select(ranks, 0, kept_indices)
    output_height, output_width, output_depth = _output_shape(self)
    values = bev_pool(
        values,
        coords,
        batch_size,
        output_depth,
        output_height,
        output_width,
        ranks=ranks,
    )
    return torch.cat(values.unbind(dim=2), 1)


def base_transform_correct_bev_pool_geometry_boundaries(
    self,
    coords,
    ranks,
    kept,
    boundary,
    inv_post_rots,
    post_trans,
    combine,
    camera2lidar_trans,
    extra_rots,
    extra_trans,
    batch_size,
):
    """Expose the optimized base.py boundary-correction class method."""

    return _correct_geometry_boundaries(
        self,
        coords,
        ranks,
        kept,
        boundary,
        inv_post_rots,
        post_trans,
        combine,
        camera2lidar_trans,
        extra_rots,
        extra_trans,
        batch_size,
    )


def base_transform_bev_pool(self, geom_feats, x):
    """Prepare geometry natively, then pool dense or factorized features."""

    import torch
    bev_pool = _resolve_bev_pool()

    prepared = isinstance(geom_feats, PreparedGeometry)
    factorized = isinstance(x, DepthFeatureFactorization)
    if factorized:
        depth, features = x
        batch_cameras, depth_bins, height, width = depth.shape
        batch = geom_feats.batch_size if prepared else geom_feats.shape[0]
        cameras = batch_cameras // batch if prepared else geom_feats.shape[1]
        if batch_cameras != batch * cameras:
            raise ValueError("depth factors do not match BEV geometry cameras")
        channels = features.shape[-1]
        expected = (batch, cameras, depth_bins, height, width)
        if not prepared and tuple(geom_feats.shape[:5]) != expected:
            raise ValueError("depth factors do not match BEV geometry shape")
    else:
        batch, cameras, depth_bins, height, width, channels = x.shape

    point_count = batch * cameras * depth_bins * height * width
    output_height = int((self.xbound[1] - self.xbound[0]) / self.xbound[2])
    output_width = int((self.ybound[1] - self.ybound[0]) / self.ybound[2])
    output_depth = int((self.zbound[1] - self.zbound[0]) / self.zbound[2])
    native_prepare = prepared or (
        use_bev_pool_prepare_opt()
        and geom_feats.is_cuda
        and geom_feats.dtype == torch.float32
    )

    if not native_prepare:
        if factorized:
            depth, image_features = x
            dense = depth.unsqueeze(-1) * image_features.unsqueeze(1)
            x = dense.view(
                batch,
                cameras,
                depth_bins,
                height,
                width,
                channels,
            )
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
        kept_indices = torch.nonzero(kept, as_tuple=False).flatten()
        values = torch.index_select(values, 0, kept_indices)
        coords = torch.index_select(coords, 0, kept_indices)
        values = bev_pool(
            values,
            coords,
            batch,
            output_depth,
            output_height,
            output_width,
        )
        return torch.cat(values.unbind(dim=2), 1)

    if prepared:
        coords, ranks, kept, prepared_batch = geom_feats
        if prepared_batch != batch:
            raise ValueError("prepared BEV geometry batch size mismatch")
    else:
        from turbo_physai.optimizations.common.mmdet3d.bev_pool import (
            bev_pool_prepare,
        )

        coords, ranks, kept = bev_pool_prepare(
            geom_feats,
            self.bx,
            self.dx,
            self.nx,
            batch,
            output_depth,
            output_height,
            output_width,
        )
    kept_indices = torch.nonzero(kept, as_tuple=False).flatten()
    coords = torch.index_select(coords, 0, kept_indices)
    ranks = torch.index_select(ranks, 0, kept_indices)
    if factorized:
        values = gather_factorized_depth_features(x, kept_indices)
    else:
        values = torch.index_select(x.reshape(point_count, channels), 0, kept_indices)

    values = bev_pool(
        values,
        coords,
        batch,
        output_depth,
        output_height,
        output_width,
        ranks=ranks,
    )
    return torch.cat(values.unbind(dim=2), 1)
