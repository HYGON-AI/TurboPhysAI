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

"""MMDetection3D BEV pooling adapters and QuickCumsum replacements."""


def quick_cumsum_forward(ctx, x, geom_feats, ranks):
    """QuickCumsum forward using explicit indices instead of mask indexing."""

    import torch

    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]
    kept_indices = torch.nonzero(kept, as_tuple=False).flatten()
    x = torch.index_select(x, 0, kept_indices)
    geom_feats = torch.index_select(geom_feats, 0, kept_indices)
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    ctx.save_for_backward(kept)
    ctx.mark_non_differentiable(geom_feats)
    return x, geom_feats


def quick_cumsum_backward(ctx, gradx, gradgeom):
    """QuickCumsum backward using index_select for the segment mapping."""

    del gradgeom
    import torch

    (kept,) = ctx.saved_tensors
    back = torch.cumsum(kept, 0)
    back -= kept.to(back.dtype)
    return torch.index_select(gradx, 0, back), None, None


def bev_pool_prepare(geom_feats, bx, dx, nx, B, D, H, W):
    from turbo_physai import operators

    return operators.bev_pool_prepare(
        geom_feats, bx, dx, nx, B, D, H, W
    )


def bev_pool_prepare_geometry(
    frustum,
    inv_post_rots,
    post_trans,
    combine,
    camera2lidar_trans,
    extra_rots,
    extra_trans,
    bx,
    dx,
    nx,
    B,
    D,
    H,
    W,
    boundary_eps=1.0e-3,
):
    from turbo_physai import operators

    return operators.bev_pool_prepare_geometry(
        frustum, inv_post_rots, post_trans, combine, camera2lidar_trans,
        extra_rots, extra_trans, bx, dx, nx, B, D, H, W, boundary_eps,
    )


def bev_pool(feats, coords, B, D, H, W, ranks=None):
    from turbo_physai import operators

    return operators.bev_pool(
        feats, coords, B, D, H, W, ranks
    )
