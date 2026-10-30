# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile the tensor-only geometry core of LSSViewTransformer."""

import functools
import os

import torch


def get_ego_coor_core(
    frustum, sensor2ego, cam2imgs_inv, post_rots_inv, post_trans, bda
):
    batch, cameras, _, _ = sensor2ego.shape
    points = frustum - post_trans.view(batch, cameras, 1, 1, 1, 3)
    points = post_rots_inv.view(batch, cameras, 1, 1, 1, 3, 3).matmul(
        points.unsqueeze(-1)
    )
    points = torch.cat(
        (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5
    )
    combine = sensor2ego[:, :, :3, :3].matmul(cam2imgs_inv)
    points = combine.view(batch, cameras, 1, 1, 1, 3, 3).matmul(
        points
    ).squeeze(-1)
    points += sensor2ego[:, :, :3, 3].view(batch, cameras, 1, 1, 1, 3)
    return bda.view(batch, 1, 1, 1, 1, 3, 3).matmul(
        points.unsqueeze(-1)
    ).squeeze(-1)


def get_ego_coor_wrapper(original, options):
    compile_kwargs = {}
    if "mode" in options:
        compile_kwargs["mode"] = options["mode"]
    compiled = get_ego_coor_core
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") != "1":
        compiled = torch.compile(get_ego_coor_core, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, sensor2ego, ego2global, cam2imgs, post_rots,
                post_trans, bda):
        if sensor2ego.device.type != "cuda":
            return original(
                self, sensor2ego, ego2global, cam2imgs, post_rots,
                post_trans, bda,
            )
        del ego2global
        return compiled(
            self.frustum.to(sensor2ego), sensor2ego,
            torch.inverse(cam2imgs), torch.inverse(post_rots), post_trans, bda,
        )

    return wrapped
