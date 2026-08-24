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

"""Contract-preserving MMDetection3D BEV pooling optimization."""


_NATIVE_BEV_POOL_AUTOGRAD = None


def _ops():
    from turbo_physai import ops

    return ops


def _native_bev_pool_autograd():
    global _NATIVE_BEV_POOL_AUTOGRAD
    if _NATIVE_BEV_POOL_AUTOGRAD is not None:
        return _NATIVE_BEV_POOL_AUTOGRAD

    import torch

    class NativeQuickCumsumCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, geom_feats, ranks, B, D, H, W):
            extension = _ops()
            kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
            kept[1:] = ranks[1:] != ranks[:-1]
            interval_starts = torch.where(kept)[0].int()
            interval_lengths = torch.zeros_like(interval_starts)
            interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
            interval_lengths[-1] = x.shape[0] - interval_starts[-1]
            geom_feats = geom_feats.int()
            output = extension.bev_pool_forward(
                x,
                geom_feats,
                interval_lengths,
                interval_starts,
                B,
                D,
                H,
                W,
            )
            ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
            ctx.saved_shapes = B, D, H, W
            return output

        @staticmethod
        def backward(ctx, output_grad):
            extension = _ops()
            interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
            B, D, H, W = ctx.saved_shapes
            x_grad = extension.bev_pool_backward(
                output_grad.contiguous(),
                geom_feats,
                interval_lengths,
                interval_starts,
                B,
                D,
                H,
                W,
            )
            return x_grad, None, None, None, None, None, None

    _NATIVE_BEV_POOL_AUTOGRAD = NativeQuickCumsumCuda
    return NativeQuickCumsumCuda


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
    """Prepare integer coordinates, ranks, and the in-range mask natively."""

    extension = _ops()
    return extension.bev_pool_prepare(
        geom_feats.contiguous(),
        bx.contiguous(),
        dx.contiguous(),
        nx.contiguous(),
        int(B),
        int(D),
        int(H),
        int(W),
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
    """Fuse camera-to-BEV geometry and return boundary candidates."""

    extension = _ops()
    return extension.bev_pool_prepare_geometry(
        frustum.contiguous(),
        inv_post_rots.contiguous(),
        post_trans.contiguous(),
        combine.contiguous(),
        camera2lidar_trans.contiguous(),
        extra_rots.contiguous(),
        extra_trans.contiguous(),
        bx.contiguous(),
        dx.contiguous(),
        nx.contiguous(),
        int(B),
        int(D),
        int(H),
        int(W),
        float(boundary_eps),
    )


def bev_pool(feats, coords, B, D, H, W, ranks=None):
    """Sort once and call TurboPhysAI's verified bundled BEV-pool extension."""

    import torch

    if feats.shape[0] != coords.shape[0]:
        raise ValueError(
            "TurboPhysAI BEVFusion bev_pool requires equal feature/coord rows"
        )
    if ranks is None:
        ranks = (
            coords[:, 0] * (W * D * B)
            + coords[:, 1] * (D * B)
            + coords[:, 2] * B
            + coords[:, 3]
        )
    indices = ranks.argsort()
    feats = torch.index_select(feats, 0, indices)
    coords = torch.index_select(coords, 0, indices)
    ranks = torch.index_select(ranks, 0, indices)
    output = _native_bev_pool_autograd().apply(
        feats, coords, ranks, B, D, H, W
    )
    return output.permute(0, 4, 1, 2, 3).contiguous()
