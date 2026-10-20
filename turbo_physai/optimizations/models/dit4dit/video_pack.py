# Copyright 2025 DiT4DiT team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Adapted from DiT4DiT 3a0c3c5; see THIRD_PARTY_NOTICES.md.

"""Video preparation on normalized training tensors; one scoped compile wrapper."""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import torch


def pack_flow_matching_video(videos_bcthw, gt_bcthw):
    # Upstream fallback accepts either [0,1] or [-1,1] GT.
    if gt_bcthw.min() >= 0:
        gt_bcthw = gt_bcthw * 2.0 - 1.0
    return torch.cat(
        [videos_bcthw.to(device=gt_bcthw.device, dtype=gt_bcthw.dtype), gt_bcthw], dim=2
    )


def _pack_normalized(videos_bcthw, gt_bcthw):
    return torch.cat(
        [videos_bcthw.to(device=gt_bcthw.device, dtype=gt_bcthw.dtype),
         gt_bcthw * 2.0 - 1.0], dim=2
    )


def compile_pack_wrapper(original, options):
    # Returning the compiled callable lets the engine own install/rollback;
    # no mutable global compile switch or lazy singleton is required.
    return torch.compile(
        _pack_normalized, backend="inductor", fullgraph=True, dynamic=False,
    )


def to_chw_float01(
        x, *, assume_normalized_float: bool = False
    ) -> torch.Tensor:
        """
        Accept PIL / numpy(HWC) / torch(CHW or HWC) and return torch float tensor in CHW, normalized to [0,1].
        """
        if isinstance(x, torch.Tensor):
            # The training dataset guarantees normalized floating-point video
            # tensors. Only that explicit path stays on the accelerator. Other
            # callers retain the original CPU conversion and /255 behavior,
            # including its bit-level rounding.
            device_fast_path = assume_normalized_float and x.is_floating_point()
            t = x.detach() if device_fast_path else x.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):  # CHW
                if t.shape[0] == 4:
                    t = t[:3]
            elif t.ndim == 3 and t.shape[-1] in (1, 3, 4):  # HWC
                if t.shape[-1] == 4:
                    t = t[..., :3]
                t = t.permute(2, 0, 1).contiguous()
            else:
                raise ValueError(f"Unsupported torch image shape {tuple(t.shape)}; expected CHW or HWC.")
            t = t.float()
            if not device_fast_path and t.max() > 1.0:
                t = t / 255.0
            return t.contiguous()

        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:
                arr = np.repeat(arr[:, :, None], 3, axis=2)
            if arr.ndim != 3:
                raise ValueError(f"Expected numpy image with shape (H,W,C), got {arr.shape}")
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.shape[2] != 3:
                raise ValueError(f"Expected RGB image, got shape={arr.shape}")
            t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
            if t.max() > 1.0:
                t = t / 255.0
            return t.contiguous()

        # PIL.Image.Image or compatible
        arr = np.array(x, dtype=np.uint8)  # (H,W,3)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape={arr.shape}")
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0
        return t.contiguous()

def build_cosmos_inputs(
        self,
        images: Sequence[Sequence[Any]],
        instructions: Sequence[str],
        **kwargs,
    ) -> dict:
        """
        Convert multi-frame images + instruction into Cosmos inputs.

        `images` is a list of lists: images[b] = [frame_0, frame_1, ..., frame_T-1]
        where frames are ordered temporally after pixel-space downsampling.

        Frame 0 is used as the conditioning frame (videos), and the remaining
        frames are future frames for video flow-matching supervision (future_videos).
        """

        if len(images) != len(instructions):
            raise ValueError("images and instructions must have the same batch size")

        all_cond = []      # frame 0 per sample
        all_future = []    # frames 1..T-1 per sample
        n_future = None

        for views in images:
            if not isinstance(views, (list, tuple)) or len(views) == 0:
                raise ValueError("Each sample must provide at least 1 view image")

            chw_frames = [
                to_chw_float01(v, assume_normalized_float=True)
                for v in views
            ]

            all_cond.append(chw_frames[0])
            current_n_future = len(chw_frames) - 1
            if n_future is None:
                n_future = current_n_future
            elif current_n_future != n_future:
                raise ValueError(
                    "All samples must provide the same number of future frames"
                )
            if current_n_future:
                future = torch.stack(chw_frames[1:], dim=0)  # (T_f, C, H, W)
                all_future.append(future)

        # videos (condition): (B, T=1, C, H, W)
        videos = torch.stack(all_cond, dim=0).unsqueeze(1)
        height, width = int(videos.shape[-2]), int(videos.shape[-1])

        # future_videos: (B, T_f, C, H, W) or None
        future_videos = None
        if len(all_future) > 0:
            future_videos = torch.stack(all_future, dim=0)  # (B, T_f, C, H, W)

        return {
            "prompts": list(instructions),
            "videos": videos,
            "height": height,
            "width": width,
            "future_videos": future_videos,
        }
