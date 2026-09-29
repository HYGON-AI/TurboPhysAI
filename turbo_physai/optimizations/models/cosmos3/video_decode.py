# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""Platform-aware video decoding helpers for training data."""

from __future__ import annotations

import os

# Load the real package before the import-replacement handler links this module.
import cosmos_framework.utils
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

_MAX_VIDEO_FRAMES = 32
_TARGET_VIDEO_FPS = 2.0


def is_hcu_runtime() -> bool:
    """Return whether the current PyTorch runtime is the HIP-compatible HCU path."""
    if getattr(torch.version, "hip", None):
        return True
    return bool(os.environ.get("HIP_VISIBLE_DEVICES")) and not getattr(torch.version, "cuda", None)


def get_video_decoder_class() -> Any:
    try:
        from torchcodec.decoders import VideoDecoder
    except (ImportError, OSError, RuntimeError, TypeError, AttributeError) as exc:
        raise ImportError("TorchCodec is required for Cosmos3 video decoding. Install the torchcodec package.") from exc
    return VideoDecoder


def get_audio_decoder_class() -> Any:
    try:
        from torchcodec.decoders import AudioDecoder
    except (ImportError, OSError, RuntimeError, TypeError, AttributeError) as exc:
        raise ImportError("TorchCodec is required for Cosmos3 audio decoding. Install the torchcodec package.") from exc
    return AudioDecoder


def create_video_decoder(source: Any, *args: Any, **kwargs: Any) -> Any:
    decoder_cls = get_video_decoder_class()
    return decoder_cls(source, *args, **kwargs)


def create_audio_decoder(source: Any, *args: Any, **kwargs: Any) -> Any:
    decoder_cls = get_audio_decoder_class()
    return decoder_cls(source, *args, **kwargs)


def probe_has_audio_stream(source: Any) -> bool:
    """Probe whether the video/audio container has an audio stream."""
    try:
        from torchcodec._core import create_from_bytes, get_container_metadata

        if isinstance(source, (bytes, bytearray, memoryview)):
            handle = create_from_bytes(bytes(source))
            meta = get_container_metadata(handle)
            has_audio = meta.best_audio_stream_index is not None
            del handle, meta
            return has_audio
    except (ImportError, AttributeError, OSError, RuntimeError):
        pass
    return False


def _make_video_decoder(source: Any) -> Any:
    return create_video_decoder(source)


def decode_video_to_pil_frames(
    video_bytes: bytes,
    *,
    max_frames: int = _MAX_VIDEO_FRAMES,
    target_fps: float = _TARGET_VIDEO_FPS,
) -> tuple[list[Any], float]:
    """Decode and sample a byte video into PIL RGB frames using TorchCodec."""

    from PIL import Image

    decoder = _make_video_decoder(video_bytes)
    total_frames = int(decoder.metadata.num_frames or 0)
    source_fps = float(decoder.metadata.average_fps or 0.0) or 30.0
    if total_frames <= 0:
        raise ValueError("video has zero frames")

    stride = max(1, int(round(source_fps / target_fps)))
    indices = list(range(0, total_frames, stride))
    if len(indices) > max_frames:
        step = len(indices) / max_frames
        indices = [indices[int(index * step)] for index in range(max_frames)]

    frames_tensor = decoder.get_frames_at(indices=indices).data
    frames_np = frames_tensor.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames_np]
    effective_fps = source_fps / stride if stride > 0 else source_fps
    return frames, float(effective_fps)


def install_torchcodec_pyav_fallback() -> bool:
    """Deprecated no-op: TorchCodec is used directly without PyAV fallback."""
    return False
