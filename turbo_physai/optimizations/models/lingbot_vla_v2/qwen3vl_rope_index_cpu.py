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

"""在 host 侧算 Qwen3-VL 的 mRoPE 位置索引，替掉上游逐标量同步的写法。"""

from __future__ import annotations

import torch


_TEMPLATE_CACHE: dict = {}


def _grid_template(grid_t: int, grid_h: int, grid_w: int):
    """展开成与 arange/expand/flatten 相同的 t/h/w 序列：t 最外、w 最内。"""
    key = (grid_t, grid_h, grid_w)
    cached = _TEMPLATE_CACHE.get(key)
    if cached is not None:
        return cached
    t_list = [t for t in range(grid_t) for _ in range(grid_h) for _ in range(grid_w)]
    h_list = [h for _ in range(grid_t) for h in range(grid_h) for _ in range(grid_w)]
    w_list = [w for _ in range(grid_t) for _ in range(grid_h) for w in range(grid_w)]
    cached = (t_list, h_list, w_list)
    _TEMPLATE_CACHE[key] = cached
    return cached


def get_rope_index(
    self,
    input_ids=None,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
):
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        attention_mask = attention_mask.to(total_input_ids.device)

        device = total_input_ids.device
        dtype = total_input_ids.dtype
        n_batch, seq_len = total_input_ids.shape

        ids_cpu = total_input_ids.detach().to("cpu").tolist()
        mask_cpu = attention_mask.detach().to("cpu").tolist()
        img_cpu = image_grid_thw.detach().to("cpu").tolist() if image_grid_thw is not None else None
        vid_cpu = video_grid_thw.detach().to("cpu").tolist() if video_grid_thw is not None else None

        pos_t = [[1] * seq_len for _ in range(n_batch)]
        pos_h = [[1] * seq_len for _ in range(n_batch)]
        pos_w = [[1] * seq_len for _ in range(n_batch)]

        image_index = 0
        video_index = 0
        mrope_position_deltas = []

        for i in range(n_batch):
            valid = [j for j, m in enumerate(mask_cpu[i]) if m == 1]
            tokens = [ids_cpu[i][j] for j in valid]
            n_tok = len(tokens)

            starts = [p for p, tok in enumerate(tokens) if tok == vision_start_token_id]
            vision_tokens = [tokens[p + 1] for p in starts if p + 1 < n_tok]
            image_nums = sum(1 for tok in vision_tokens if tok == image_token_id)
            video_nums = sum(1 for tok in vision_tokens if tok == video_token_id)

            llm_t: list = []
            llm_h: list = []
            llm_w: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            cur_max = None

            for _ in range(image_nums + video_nums):
                if remain_images > 0 and image_token_id in tokens:
                    ed_image = tokens.index(image_token_id, st)
                else:
                    ed_image = n_tok + 1
                if remain_videos > 0 and video_token_id in tokens:
                    ed_video = tokens.index(video_token_id, st)
                else:
                    ed_video = n_tok + 1

                if ed_image < ed_video:
                    t, h, w = img_cpu[image_index]
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = vid_cpu[video_index]
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                grid_t = int(t)
                grid_h = int(h) // spatial_merge_size
                grid_w = int(w) // spatial_merge_size
                text_len = ed - st

                st_idx = 0 if cur_max is None else cur_max + 1
                if text_len > 0:
                    llm_t.extend(range(st_idx, st_idx + text_len))
                    llm_h.extend(range(st_idx, st_idx + text_len))
                    llm_w.extend(range(st_idx, st_idx + text_len))
                    cur_max = st_idx + text_len - 1

                base = text_len + st_idx
                t_list, h_list, w_list = _grid_template(grid_t, grid_h, grid_w)
                llm_t.extend([v + base for v in t_list])
                llm_h.extend([v + base for v in h_list])
                llm_w.extend([v + base for v in w_list])
                if grid_t and grid_h and grid_w:
                    cur_max = max(cur_max, max(grid_t - 1, grid_h - 1, grid_w - 1) + base)

                st = ed + grid_t * grid_h * grid_w

            if st < n_tok:
                st_idx = 0 if cur_max is None else cur_max + 1
                text_len = n_tok - st
                llm_t.extend(range(st_idx, st_idx + text_len))
                llm_h.extend(range(st_idx, st_idx + text_len))
                llm_w.extend(range(st_idx, st_idx + text_len))
                if text_len > 0:
                    cur_max = st_idx + text_len - 1

            if not (len(llm_t) == len(llm_h) == len(llm_w) == len(valid)):
                raise RuntimeError(
                    "get_rope_index: position id length mismatch "
                    f"(ids={len(llm_t)}, mask={len(valid)}) at sample {i}"
                )
            for j, v in zip(valid, llm_t):
                pos_t[i][j] = v
            for j, v in zip(valid, llm_h):
                pos_h[i][j] = v
            for j, v in zip(valid, llm_w):
                pos_w[i][j] = v

            mrope_position_deltas.append(
                max(max(llm_t), max(llm_h), max(llm_w)) + 1 - seq_len
            )

        position_ids = torch.tensor([pos_t, pos_h, pos_w], device=device, dtype=dtype)
        deltas = torch.tensor(mrope_position_deltas, device=device).unsqueeze(1)
        return position_ids, deltas

    if attention_mask is not None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
    else:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .view(1, 1, -1)
            .expand(3, input_ids.shape[0], -1)
        )
        mrope_position_deltas = torch.zeros(
            [input_ids.shape[0], 1],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

    return position_ids, mrope_position_deltas
