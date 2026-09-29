# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Channels-last helpers enabled automatically on the HIP training path.

The logical tensor shapes stay channel-first (NCHW/NCDHW).  The helpers only
change the physical stride layout, so model/checkpoint interfaces do not need
to change.  HIP builds enable this path automatically; non-HIP builds keep
the original memory format.
"""

from __future__ import annotations

import torch
from torch import nn


def training_channels_last_enabled() -> bool:
    """Whether the HIP training NHWC/NDHWC path is enabled."""

    return bool(getattr(torch.version, "hip", None))


def to_channels_last(tensor: torch.Tensor) -> torch.Tensor:
    """Return *tensor* in channels-last format on HIP builds.

    Four-dimensional tensors use ``channels_last`` (NHWC physical layout),
    five-dimensional tensors use ``channels_last_3d`` (NDHWC physical layout).
    Rank-3 token tensors are intentionally unchanged: they have no NHWC
    equivalent without changing the model's logical representation.
    """

    if not training_channels_last_enabled():
        return tensor
    if tensor.dim() == 4:
        return tensor.contiguous(memory_format=torch.channels_last)
    if tensor.dim() == 5:
        return tensor.contiguous(memory_format=torch.channels_last_3d)
    return tensor


def apply_channels_last_conv_weights(module: nn.Module) -> tuple[int, int]:
    """Convert every Conv2d/Conv3d weight below *module* when enabled.

    This is called while the Generator network is still on ``meta`` and before
    FSDP sharding.  ``Module.to(memory_format=...)`` changes parameter strides
    without changing logical shapes or values.

    Returns:
        ``(conv2d_count, conv3d_count)`` converted by this call.
    """

    if not training_channels_last_enabled():
        return 0, 0

    conv2d_count = 0
    conv3d_count = 0
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            child.to(memory_format=torch.channels_last)
            conv2d_count += 1
        elif isinstance(child, nn.Conv3d):
            child.to(memory_format=torch.channels_last_3d)
            conv3d_count += 1
    return conv2d_count, conv3d_count
