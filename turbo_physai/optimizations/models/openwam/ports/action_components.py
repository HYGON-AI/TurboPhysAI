# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.norms import lightop_rms_norm


def RMSNorm_forward(self, x: torch.Tensor) -> torch.Tensor:
    if enabled("lightop_norm"):
        fast = lightop_rms_norm(x, self.weight, self.eps)
        if fast is not None:
            return fast
    dtype = x.dtype
    normed = x.float() * torch.rsqrt(
        x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
    )
    return normed.to(dtype) * self.weight


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

RMSNorm_forward_wrapper = _port_wrapper(
    RMSNorm_forward,
    "openwam.model.action_backbone.components.RMSNorm.forward",
    "openwam.compute",
)
