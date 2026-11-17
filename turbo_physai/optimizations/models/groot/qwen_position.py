# Adapted from Isaac-GR00T, Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import torch
from functools import wraps

def _grid_thw_as_int_rows(grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
    """Materialize small vision-grid metadata on CPU with at most one DtoH copy."""
    grid_cpu = grid_thw.detach().to(device="cpu", dtype=torch.long)
    return [tuple(row) for row in grid_cpu.tolist()]


def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Build interpolation metadata on CPU, then do the trainable lookup on-device.

    The Transformers implementation already builds ``idx_list`` and ``weight_list``
    on CPU. When ``grid_thw`` is on an accelerator, however, using its scalar values
    as ``linspace``/``repeat``/``view`` arguments performs many synchronous DtoH
    copies. Converting the tiny grid once keeps the same arithmetic and autograd path.
    """
    grid_rows = _grid_thw_as_int_rows(grid_thw)
    weight = self.pos_embed.weight
    cache_key = (tuple(grid_rows), weight.device, weight.dtype, self.num_grid_per_side)
    cache = getattr(self, "_gr00t_fast_pos_cache", None)
    if cache is not None and cache[0] == cache_key and not torch.jit.is_tracing():
        idx_tensor, weight_tensor = cache[1:]
    else:
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _t, h, w in grid_rows:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device="cpu")
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device="cpu")

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

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
            for index in range(4):
                idx_list[index].extend(indices[index].tolist())
                weight_list[index].extend(weights[index].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=weight.device)
        weight_tensor = torch.tensor(weight_list, dtype=weight.dtype, device=weight.device)
        if not torch.jit.is_tracing():
            self._gr00t_fast_pos_cache = (cache_key, idx_tensor, weight_tensor)

    pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for _t, h, w in grid_rows])
    merge_size = self.config.spatial_merge_size
    permuted = []
    for pos_embed, (t, h, w) in zip(patch_pos_embeds, grid_rows):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        permuted.append(pos_embed)
    return torch.cat(permuted)


def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
    """Build integer rotary coordinates on CPU and transfer them in one batch."""
    grid_rows = _grid_thw_as_int_rows(grid_thw)
    max_hw = max(max(height, width) for _t, height, width in grid_rows)
    freq_table = self.rotary_pos_emb(max_hw)
    merge_size = self.spatial_merge_size
    cache_key = (tuple(grid_rows), freq_table.device, merge_size)
    cache = getattr(self, "_gr00t_rot_pos_cache", None)
    if cache is not None and cache[0] == cache_key and not torch.jit.is_tracing():
        pos_ids_device = cache[1]
    else:
        pos_ids = []
        for num_frames, height, width in grid_rows:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device="cpu")
            block_cols = torch.arange(merged_w, device="cpu")
            intra_row = torch.arange(merge_size, device="cpu")
            intra_col = torch.arange(merge_size, device="cpu")
            row_idx = (
                block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            )
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            pos_ids.append(coords)

        pos_ids_device = torch.cat(pos_ids).to(freq_table.device)
        if not torch.jit.is_tracing():
            self._gr00t_rot_pos_cache = (cache_key, pos_ids_device)
    return freq_table[pos_ids_device].flatten(1)


def rope_index_wrapper(original, group_options):
    @wraps(original)
    def wrapped(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        if input_ids is None or (image_grid_thw is None and video_grid_thw is None):
            return original(self, input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask)
        device = input_ids.device
        cpu = lambda value: None if value is None else value.detach().to("cpu")
        positions, deltas = original(self, cpu(input_ids), cpu(image_grid_thw), cpu(video_grid_thw), attention_mask=cpu(attention_mask))
        return positions.to(device), deltas.to(device)
    return wrapped
