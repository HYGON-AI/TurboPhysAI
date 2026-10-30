# Portions are adapted from the PointPillars HCU optimization project:
# Source: PointPillars/mmdetection3d/mmdet3d/structures/bbox_3d/utils.py
# Revision: bd37518d0408bcea4ac879a90256ffb8b8628258
# Project license: Apache-2.0
# Project copyright: Copyright 2018-2019 Open-MMLab. All rights reserved.
#
# Portions are derived from OpenMMLab MMDetection3D:
# https://github.com/open-mmlab/mmdetection3d
# File: mmdet3d/structures/bbox_3d/utils.py
# Commit: fe25f7a51d36e3702f961e198894580d83c4387b
# Copyright (c) OpenMMLab. All rights reserved.
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-native limit_period wrapper used by compiled PointPillars paths."""

from functools import wraps

import numpy as np
import torch


def native_limit_period_wrapper(original, options):
    """Bypass array_converter for tensors while preserving NumPy behavior."""
    @wraps(original)
    def wrapped(val, offset=0.5, period=np.pi):
        if isinstance(val, torch.Tensor):
            return val - torch.floor(val / period + offset) * period
        if isinstance(val, np.ndarray):
            return val - np.floor(val / period + offset) * period
        return original(val, offset, period)

    return wrapped
