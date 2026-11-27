# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""2x bilinear 上采样布局优化：替换 aten 的 upsample，绕开 inductor 的 gather 展开。"""

from __future__ import annotations

import torch
import torch.nn as nn


def _is_target(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Upsample)
        and module.mode == "bilinear"
        and module.align_corners is False
        and module.recompute_scale_factor is None
        and module.size is None
        and module.scale_factor == 2
    )


@torch.compiler.disable
def _upsample_bilinear_2x(x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    xc = x.contiguous()
    if x.dtype in (torch.float16, torch.bfloat16):
        upsampled = torch.ops.aten.upsample_bilinear2d.default(
            xc.float(), [2 * h, 2 * w], False, None, None
        )
        return upsampled.to(x.dtype)
    return torch.ops.aten.upsample_bilinear2d.default(
        xc, [2 * h, 2 * w], False, None, None
    )


def upsample_forward(original, options):

    def forward(self, input):
        if (
            _is_target(self)
            and isinstance(input, torch.Tensor)
            and input.dim() == 4
            and input.dtype is torch.float32
            and input.numel() > 0
        ):
            return _upsample_bilinear_2x(input)
        return original(self, input)

    return forward
