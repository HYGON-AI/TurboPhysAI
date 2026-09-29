# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import cosmos_framework.data.generator.augmentors.interleaved_video_parsing as _source
from ._source_binding import bind_missing_globals
from .video_decode import create_video_decoder
_SUPPORTS_VIDEO_DECODER_TRANSFORMS = None
_WARNED_POST_DECODE_TRANSFORMS = False

def _create_video_decoder(
    video: bytes,
    seek_mode: str,
    num_ffmpeg_threads: int,
    transforms: _PostDecodeTransforms = None,
) -> tuple[Any, _PostDecodeTransforms]:
    global _SUPPORTS_VIDEO_DECODER_TRANSFORMS, _WARNED_POST_DECODE_TRANSFORMS

    kwargs = {"seek_mode": seek_mode, "num_ffmpeg_threads": num_ffmpeg_threads}
    if transforms is None:
        return create_video_decoder(video, **kwargs), None

    if _SUPPORTS_VIDEO_DECODER_TRANSFORMS is not False:
        try:
            decoder = create_video_decoder(video, transforms=transforms, **kwargs)
            _SUPPORTS_VIDEO_DECODER_TRANSFORMS = True
            return decoder, None
        except TypeError as e:
            if "transforms" not in str(e):
                raise
            _SUPPORTS_VIDEO_DECODER_TRANSFORMS = False

    if not _WARNED_POST_DECODE_TRANSFORMS:
        log.warning(
            "Installed torchcodec does not support VideoDecoder(transforms=...); "
            "applying video transforms after frame decode.",
            rank0_only=False,
        )
        _WARNED_POST_DECODE_TRANSFORMS = True
    return create_video_decoder(video, **kwargs), transforms

def _probe_video_len(self, video: bytes) -> int:
    video_decoder = create_video_decoder(
        video,
        seek_mode=self.seek_mode,
        num_ffmpeg_threads=self.video_decode_num_threads,
    )
    try:
        return len(video_decoder)
    finally:
        del video_decoder


bind_missing_globals(globals(), _source)
