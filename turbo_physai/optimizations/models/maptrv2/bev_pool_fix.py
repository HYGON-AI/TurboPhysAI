# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/maptr/modules/encoder.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Copyright (c) 2022 Hust Vision Lab
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""ROCm bev_pool layout fix for MapTRv2.

ROCm/HIP bev_pool returns [B, Z, H, W, C] (channels-last) instead of
[B, C, Z, H, W] as the CUDA version does.  The collapse-Z cat that follows
therefore receives the wrong axis and produces a misshapen tensor, causing the
downstream Conv2d to raise a channel-count mismatch.

The replacement returns the baseline's collapsed 4D layout by default. The
reference-boundaries path requests the uncollapsed 5D layout through
``return_5d=True`` so both Groups use the same pooling implementation.

Torch and the native pooling operator are loaded only when pooling is called.
"""


def normalize_bev_pool_layout(output, batch, channels, nx, *, channels_last):
    """Validate and normalize the two supported 5D BEV pooling layouts."""
    depth, height, width = int(nx[2]), int(nx[0]), int(nx[1])
    expected_nchw = (batch, channels, depth, height, width)
    expected_nhwc = (batch, depth, height, width, channels)
    shape = tuple(output.shape)
    is_nchw = shape == expected_nchw
    is_nhwc = shape == expected_nhwc
    if is_nchw and is_nhwc:
        raise RuntimeError(
            f"ambiguous BEV pooling layout for shape {shape}: "
            "both [B,C,Z,H,W] and [B,Z,H,W,C] match"
        )
    if not (is_nchw or is_nhwc):
        raise RuntimeError(
            f"invalid BEV pooling shape {shape}: expected "
            f"[B,C,Z,H,W]={expected_nchw} or [B,Z,H,W,C]={expected_nhwc}"
        )
    if channels_last and is_nchw:
        return output.permute(0, 2, 3, 4, 1).contiguous()
    if not channels_last and is_nhwc:
        return output.permute(0, 4, 1, 2, 3).contiguous()
    return output


def _bev_pool(*args):
    from mmdet3d.ops import bev_pool

    return bev_pool(*args)


def base_transform_bev_pool(self, geom_feats, x, *, return_5d=False):
    import torch

    B, N, D, H, W, C = x.shape
    Nprime = B * N * D * H * W

    # flatten x
    x = x.reshape(Nprime, C)

    # flatten indices
    geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
    geom_feats = geom_feats.view(Nprime, 3)
    batch_ix = torch.cat(
        [
            torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
            for ix in range(B)
        ]
    )
    geom_feats = torch.cat((geom_feats, batch_ix), 1)

    # filter out points that are outside box
    kept = (
        (geom_feats[:, 0] >= 0)
        & (geom_feats[:, 0] < self.nx[0])
        & (geom_feats[:, 1] >= 0)
        & (geom_feats[:, 1] < self.nx[1])
        & (geom_feats[:, 2] >= 0)
        & (geom_feats[:, 2] < self.nx[2])
    )
    x = x[kept]
    geom_feats = geom_feats[kept]

    x = _bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

    # The reference boundary keeps Z separate until its compiled down_sample
    # helper. The standalone replacement retains the baseline's 4D contract.
    x = normalize_bev_pool_layout(x, B, C, self.nx, channels_last=return_5d)
    if return_5d:
        return x

    # collapse Z
    final = torch.cat(x.unbind(dim=2), 1)
    return final


# BaseTransform.forward can ask this replacement for the 5D layout when the
# reference boundaries Group is active. Other bev_pool implementations fall
# back to reference._bev_pool_5d.
base_transform_bev_pool._turbo_physai_returns_5d = True
