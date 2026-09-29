# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import cosmos_framework.data.generator.reasoner.video_decoder_qwen as _source
from ._source_binding import bind_missing_globals
from .video_decode import create_video_decoder

def _video_decoder_qwen_func(
    key: str,
    data: bytes,
    processor: Qwen3VLProcessor,
    min_fps_thres: int = 4,
    max_fps_thres: int = 60,
    target_fps: float = 2.0,
    min_video_token_length: int = 16,
    max_video_token_length: int = 8192,
    num_threads: int = 0,
    random_augmentation: bool = False,
    fps_random_range: list[float] = [0.5, 1.5],
    max_video_token_length_random_range: list[float] = [0.75, 1.25],
    frame_count_random_range: Optional[list[int]] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    decoding_timeout: int = 60,
    **kwargs,
) -> dict | None:
    """Actual video decoder function.

    Args:
        key (str): Video file name/key
        data (bytes): Video binary data
        min_fps_thres (int, optional): Minimum FPS threshold. Defaults to 4.
        max_fps_thres (int, optional): Maximum FPS threshold. Defaults to 60.
        target_fps (float, optional): Target FPS. Defaults to 2.0.
        min_video_token_length (int, optional): Minimum token length. Defaults to 16.
        max_video_token_length (int, optional): Maximum token length. Defaults to 8192.
        num_threads (int, optional): Number of threads for the torchcodec video decoder. Defaults to 0.
        random_augmentation (bool, optional): Whether to randomize the FPS and max_video_token_length. Defaults to False.
        fps_random_range (list[float], optional): Random FPS range. Defaults to [10.0, 24.0].
        max_video_token_length_random_range (list[float], optional): Random max_video_token_length range. Defaults to [0.75, 1.25].
        frame_count_random_range (list[int], optional): Random frame count range. If provided, take priority over fps_random_range.
        start_frame (Optional[int], optional): Start frame. Defaults to None. If both start_frame and end_frame are provided, the video will be decoded from start_frame to end_frame.
        end_frame (Optional[int], optional): End frame. Defaults to None. If both start_frame and end_frame are provided, the video will be decoded from start_frame to end_frame.
        decoding_timeout (int, optional): Timeout in seconds. Defaults to 60.
    Raises:
        ValueError: Video fps lower than 1, skipping
        ValueError: Video fps lower than min_fps_thres, skipping
        ValueError: Video fps higher than max_fps_thres, skipping

    Returns:
        dict | None: Dictionary with video frames tensor and target FPS
    """
    # Check video extension
    extension = re.sub(r".*[.]", "", key)
    if extension.lower() not in _VIDEO_EXTENSIONS:
        return None

    # Read video with torchcodec
    video_reader = create_video_decoder(data, num_ffmpeg_threads=num_threads)
    total_frames = video_reader.metadata.num_frames
    video_fps = video_reader.metadata.average_fps

    # torchcodec returns ``None`` for containers that don't store frame count
    # or average fps (e.g. some MKV/WebM streams).  Downstream arithmetic
    # (``total_frames - 1``, ``video_fps < 1``, ...) would TypeError on None;
    # surface a ValueError so the dataloader's skip path handles it uniformly.
    if total_frames is None or video_fps is None:
        raise ValueError(f"torchcodec missing metadata (num_frames={total_frames}, average_fps={video_fps}), skipping")

    if start_frame is not None and end_frame is not None:
        total_frames = end_frame - start_frame

    if video_fps < 1:
        raise ValueError("Video fps lower than 1, skipping")
    if video_fps < min_fps_thres:
        raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
    if video_fps > max_fps_thres:
        raise ValueError(f"Video fps {video_fps} higher than {max_fps_thres}, skipping")

    if random_augmentation:
        if frame_count_random_range is not None:
            # Random number of frames
            min_frames_range, max_frames_range = frame_count_random_range
            min_frames_range = min(min_frames_range, total_frames)
            max_frames_range = min(max_frames_range, total_frames)
            target_frames = random.uniform(min_frames_range, max_frames_range)
            target_fps = target_frames / total_frames * video_fps
        else:
            # randomize fps
            target_fps = (
                random.uniform(fps_random_range[0], fps_random_range[1]) * target_fps
                if random.random() < 0.5
                else target_fps
            )
        # randomize max_video_token_length
        max_video_token_length = int(
            random.uniform(max_video_token_length_random_range[0], max_video_token_length_random_range[1])
            * max_video_token_length
        )
        log.debug(f"random_augmentation: max_video_token_length: {max_video_token_length}, target_fps: {target_fps}")

    patch_size = processor.patch_size
    min_height_width = processor.min_height_width
    temporal_patch_size = processor.temporal_patch_size
    merge_size = processor.merge_size
    min_pixels: int = token_to_pixels(min_video_token_length, patch_size, temporal_patch_size, merge_size)
    max_pixels: int = token_to_pixels(max_video_token_length, patch_size, temporal_patch_size, merge_size)
    max_frames: int = max_pixels // (min_height_width) ** 2 // temporal_patch_size

    # sample based on target fps
    nframes = smart_nframes(dict(fps=target_fps), total_frames=total_frames, video_fps=video_fps)
    nframes = min(nframes, max_frames)
    if start_frame is not None and end_frame is not None:
        idx = torch.linspace(start_frame, end_frame - 1, nframes).round().long().tolist()  # [nframes]
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()  # [nframes]

    def _decode_video() -> torch.Tensor:
        return video_reader.get_frames_at(indices=idx).data  # [T, C, H, W] uint8

    # Use ThreadPoolExecutor to run video decoding with a timeout.
    # If the thread is stuck, abandon it immediately.
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_decode_video)
    try:
        video_frames = future.result(timeout=decoding_timeout)
        executor.shutdown(wait=False)
    except TimeoutError as e:
        log.warning(f"[{key}] Video decoding timed out after {decoding_timeout} seconds")
        executor.shutdown(wait=False)
        return None

    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    # recompute max_pixels based on number of sampled frames
    nframes, _, height, width = video_frames.shape
    max_pixels = max_pixels // nframes
    if processor.use_smart_resize:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_size * merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        log.debug(
            f"resized_height: {resized_height}, resized_width: {resized_width} | original height: {height}, original width: {width}"
        )
        video_frames = transforms.functional.resize(
            video_frames,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()  # [T,C,H,W]
    video_frames = video_frames.permute(1, 0, 2, 3)  # [C,T,H,W]

    return dict(videos=video_frames, fps=sample_fps)


bind_missing_globals(globals(), _source)
