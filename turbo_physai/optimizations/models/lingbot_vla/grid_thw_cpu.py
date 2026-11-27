# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
让 grid_thw 留在 CPU 上算 window/rotary 元数据，再把结果搬到 GPU。

embed_image 每步都会构造 [1, h, w] 的 grid_thw；放在 GPU 上时 preprcess_grid_thw 里的 rot_pos_emb / get_window_index 要逐标量迭代，
每步都产生多次 device 同步。
"""

from __future__ import annotations

import functools

import torch


def _target_device(grid_thw: torch.Tensor) -> torch.device:
    if grid_thw.device.type != "cpu":
        return grid_thw.device
    return torch.device("cuda", torch.cuda.current_device())


def wrap_embed_image(original, options=None):
    @functools.wraps(original)
    def wrapped(self, image, patch_size=14, temporal_patch_size=2):
        h = w = int(image.shape[1] ** 0.5)
        image_grid_thw = torch.tensor([[1, h, w]] * image.shape[0])
        return self.get_image_features(image, image_grid_thw=image_grid_thw)

    return wrapped


def wrap_preprocess_grid_thw(original, options=None):
    @functools.wraps(original)
    def wrapped(self, grid_thw):
        rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens = original(
            self, grid_thw
        )
        device = _target_device(grid_thw)
        return (
            rotary_pos_emb.to(device),
            window_index.to(device),
            cu_window_seqlens.to(device),
            cu_seqlens.to(device),
        )

    return torch._dynamo.disable(wrapped)
