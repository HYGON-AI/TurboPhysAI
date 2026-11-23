# Derived from Sparse4D: projects/mmdet3d_plugin/models/instance_bank.py
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

import numpy as np
import torch

def get(self, batch_size, metas=None, dn_metas=None):
    instance_feature = torch.tile(
        self.instance_feature[None], (batch_size, 1, 1)
    )
    anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

    if (
        self.cached_anchor is not None
        and batch_size == self.cached_anchor.shape[0]
    ):
        history_time = self.metas["timestamp"]
        time_interval = metas["timestamp"] - history_time
        time_interval = time_interval.to(dtype=instance_feature.dtype)
        self.mask = torch.abs(time_interval) <= self.max_time_interval

        if self.anchor_handler is not None:
            T_temp2cur = (
                torch.from_numpy(
                    np.stack(
                        [
                            x["T_global_inv"]
                            @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )
                .to(dtype=self.cached_anchor.dtype)
                .to(device=self.cached_anchor.device)
            )
            self.cached_anchor = self.anchor_handler.anchor_projection(
                self.cached_anchor,
                [T_temp2cur],
                time_intervals=[-time_interval],
            )[0]

        if (
            self.anchor_handler is not None
            and dn_metas is not None
            and batch_size == dn_metas["dn_anchor"].shape[0]
        ):
            num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
            dn_anchor = self.anchor_handler.anchor_projection(
                dn_metas["dn_anchor"].flatten(1, 2),
                [T_temp2cur],
                time_intervals=[-time_interval],
            )[0]
            dn_metas["dn_anchor"] = dn_anchor.reshape(
                batch_size, num_dn_group, num_dn, -1
            )
        time_interval = torch.where(
            torch.logical_and(time_interval != 0, self.mask),
            time_interval,
            self.default_time_interval,
            # time_interval.new_tensor(self.default_time_interval),
        )
    else:
        self.reset()
        time_interval = instance_feature.new_tensor(
            [self.default_time_interval] * batch_size
        )

    return (
        instance_feature,
        anchor,
        self.cached_feature,
        self.cached_anchor,
        time_interval,
    )