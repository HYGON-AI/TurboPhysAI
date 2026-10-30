# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from turbo_physai.optimizations.models.openwam import enabled, pointwise


def rope_apply_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D rotary position embedding to Q or K.

    Mirrors FastWAM/Wan RoPE numerics by doing the complex multiply through a
    float64 complex view, then restoring the caller's dtype.

    Args:
        x:     (B, H, S, D) head-split tensor.
        freqs: (S, D // 2) complex frequencies from ``precompute_freqs_cis_1d``.

    Returns:
        Rotated tensor with the same dtype and shape as ``x``.
    """
    if enabled("rope_real"):
        return pointwise.rotate(x, freqs.view(1, 1, x.shape[-2], -1))
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.to(x_c.device).view(1, 1, x_c.shape[-2], x_c.shape[-1])
    return torch.view_as_real(x_c * freqs).flatten(-2).to(x.dtype)


def RMSNorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    if enabled("pointwise_compile") or enabled("lightop_norm"):
        return pointwise.rms_norm(x, self.weight, self.eps)
    dtype = x.dtype
    normed = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
    )
    return normed.to(dtype) * self.weight


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

rope_apply_1d_wrapper = _port_wrapper(
    rope_apply_1d,
    "openwam.model.action_backbone.components.rope_apply_1d",
    "openwam.compute",
)

RMSNorm_forward_wrapper = _port_wrapper(
    RMSNorm_forward,
    "openwam.model.action_backbone.components.RMSNorm.forward",
    "openwam.compute",
)
