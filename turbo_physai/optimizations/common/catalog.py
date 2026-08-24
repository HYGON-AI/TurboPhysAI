# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Registration entry for all built-in common optimizations."""

from . import mmcv as _mmcv  # noqa: F401 - registers MMCV optimizations
from . import mmdet3d as _mmdet3d  # noqa: F401 - registers MMDetection3D optimizations
