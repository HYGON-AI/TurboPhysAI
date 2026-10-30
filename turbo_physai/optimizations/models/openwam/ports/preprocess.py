# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from openwam.model.video_backbone.wan.preprocess import (
    Image,
    np,
    preprocess_image,
)
from turbo_physai.optimizations.models.openwam import enabled


def preprocess_video(
    video, *, dtype, device, pattern="B C T H W", min_value=-1, max_value=1
):
    """List[PIL.Image] -> stacked tensor along the ``T`` axis of ``pattern``."""
    if (
        enabled("uint8_preprocess")
        and pattern == "B C T H W"
        and all(
            isinstance(frame, Image.Image) and frame.mode in ("RGB", "RGBA")
            for frame in video
        )
    ):
        # Cast to the original target dtype BEFORE normalization, preserving
        # BF16 rounding order; only the upload representation changes.
        pixels = torch.from_numpy(
            np.stack([np.asarray(frame, dtype=np.uint8) for frame in video])
        )
        if torch.device(device).type == "cuda":
            pixels = pixels.pin_memory().to(device=device, non_blocking=True)
        else:
            pixels = pixels.to(device=device)
        pixels = pixels.to(dtype=dtype)
        pixels = pixels * ((max_value - min_value) / 255) + min_value
        return pixels.permute(3, 0, 1, 2).unsqueeze(0)
    video = [
        preprocess_image(
            image, dtype=dtype, device=device, min_value=min_value, max_value=max_value
        )
        for image in video
    ]
    video = torch.stack(video, dim=pattern.index("T") // 2)
    return video


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

preprocess_video_wrapper = _port_wrapper(
    preprocess_video,
    "openwam.model.video_backbone.wan.preprocess.preprocess_video",
    "openwam.data",
)
