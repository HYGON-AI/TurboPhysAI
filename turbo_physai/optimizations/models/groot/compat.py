# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime adaptations for the unmodified official GR00T baseline."""

import logging
import os
from functools import wraps

logger = logging.getLogger(__name__)


def training_config_wrapper(original, options):
    """Adjust configuration before validation, artifact saving and model loading."""
    @wraps(original)
    def run(config):
        if config.training.tf32:
            from transformers.utils import is_torch_tf32_available

            if not is_torch_tf32_available():
                config.training.tf32 = False
                logger.info("TF32 requested but unsupported by this device; disabling TF32")
        model_path = os.environ.get("VLM_MODEL_PATH")
        if model_path:
            config.model.model_name = model_path
        return original(config)

    return run


def opencv_video_wrapper(original, options):
    """Use the opt-in CPU OpenCV decoder, otherwise preserve official decoding.

    The OpenCV branch reproduces the indexed-frame adaptation from d9bdcf9;
    torchcodec-specific decoder_kwargs have no effect on this CPU branch.
    """
    @wraps(original)
    def get_frames_by_indices(video_path, indices, decoder_kwargs=None):
        if os.environ.get("VIDEO_DECODER_BACKEND") != "opencv":
            return original(video_path, indices, decoder_kwargs=decoder_kwargs)

        import numpy as np

        requested = np.asarray(indices, dtype=np.int64)
        if requested.size == 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        if np.any(requested < 0):
            raise ValueError(f"Frame indices must be non-negative: {requested.tolist()}")

        import cv2

        capture = cv2.VideoCapture(video_path)
        frames_by_index = {}
        wanted = set(requested.tolist())
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video with OpenCV: {video_path}")
            for index in range(int(requested.max()) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise IndexError(f"Could not decode frame {index} from video {video_path}")
                if index in wanted:
                    frames_by_index[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            capture.release()
        return np.stack([frames_by_index[int(index)] for index in requested])

    return get_frames_by_indices
