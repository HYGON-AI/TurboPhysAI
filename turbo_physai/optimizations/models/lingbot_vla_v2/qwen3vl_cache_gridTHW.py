# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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

"""
缓存 Qwen3-VL vision 的 pos-embedding 索引/权重表。
"""

from __future__ import annotations

import torch


_POS_EMBED_CACHE: dict = {}
_POS_EMBED_CACHE_LIMIT = 16


def _build_pos_embed_tables(self, grid_thw: torch.Tensor, dtype: torch.dtype):
    rows = grid_thw.detach().to("cpu").tolist()
    num_grid_per_side = self.num_grid_per_side
    merge_size = self.config.spatial_merge_size

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for _t, h, w in rows:
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long)
    weight_tensor = torch.tensor(weight_list, dtype=dtype)

    split_sizes = [int(h) * int(w) for _t, h, w in rows]
    shapes = [
        (int(t), int(h) // merge_size, int(w) // merge_size, merge_size)
        for t, h, w in rows
    ]
    return idx_tensor, weight_tensor, split_sizes, shapes


@torch._dynamo.disable
def _lookup_pos_embed_tables(self, grid_thw: torch.Tensor, device, dtype):
    key = (tuple(grid_thw.detach().to("cpu").flatten().tolist()), str(device), str(dtype))
    cached = _POS_EMBED_CACHE.get(key)
    if cached is None:
        idx_tensor, weight_tensor, split_sizes, shapes = _build_pos_embed_tables(
            self, grid_thw, dtype
        )
        if len(_POS_EMBED_CACHE) >= _POS_EMBED_CACHE_LIMIT:
            _POS_EMBED_CACHE.clear()
        cached = (idx_tensor.to(device), weight_tensor.to(device), split_sizes, shapes)
        _POS_EMBED_CACHE[key] = cached
    return cached


def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
    weight = self.pos_embed.weight
    idx_tensor, weight_tensor, split_sizes, shapes = _lookup_pos_embed_tables(
        self, grid_thw, weight.device, weight.dtype
    )

    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split(split_sizes)

    patch_pos_embeds_permute = []
    for pos_embed, (t, h, w, merge_size) in zip(patch_pos_embeds, shapes):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h, merge_size, w, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    return torch.cat(patch_pos_embeds_permute)
