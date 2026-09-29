# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VideoParsing audio-chunk extraction with container audio-stream probing."""

from __future__ import annotations

import torch

import cosmos_framework.data.generator.augmentors.video_parsing as _vp_mod

from ._source_binding import bind_missing_globals


def probe_has_audio_stream(source) -> bool:
    """Probe whether the container has an audio stream (torchcodec._core)."""
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


def create_audio_decoder(source, *args, **kwargs):
    from torchcodec.decoders import AudioDecoder

    return AudioDecoder(source, *args, **kwargs)


def _extract_audio_chunk(
    self, video_bytes: bytes, video_fps: float, frame_indices: list[int]
) -> torch.Tensor | None:  # returns [C,N_audio] or None
    """
    Extract audio chunk corresponding to the given frame indices.

    Args:
        video_bytes: Raw video bytes
        video_fps: Video frames per second
        frame_indices: List of frame indices being extracted

    Returns:
        Audio tensor of shape (C, N) or None if audio extraction fails
    """
    try:
        if not probe_has_audio_stream(video_bytes):
            return None

        # Create audio decoder
        audio_decoder = create_audio_decoder(video_bytes)

        # Calculate time range for audio corresponding to video frames
        time_start = frame_indices[0] / video_fps
        time_end = (frame_indices[-1] + 1) / video_fps  # +1 to include the last frame's duration

        # Get audio samples for the specific time range
        audio_metadata = audio_decoder.metadata
        orig_sample_rate = audio_metadata.sample_rate

        audio_samples = audio_decoder.get_samples_played_in_range(start_seconds=time_start, stop_seconds=time_end)
        audio_chunk = audio_samples.data  # [C,N_orig]

        # Resample if needed
        if orig_sample_rate != self.audio_sample_rate:
            import librosa

            audio_np = audio_chunk.numpy()
            resampled_audio_np = librosa.resample(
                audio_np, orig_sr=orig_sample_rate, target_sr=self.audio_sample_rate, axis=-1
            )
            audio_chunk = torch.from_numpy(resampled_audio_np)  # [C,N_resampled]

        # Clean up audio decoder
        del audio_decoder

        return audio_chunk

    except Exception as e:
        log.warning(f"Failed to extract audio: {e}", rank0_only=False)
        return None


bind_missing_globals(globals(), _vp_mod)
