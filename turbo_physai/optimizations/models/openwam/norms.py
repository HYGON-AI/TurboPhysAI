# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

"""Optional existing Lightop training kernels, with the native path preserved."""

from functools import lru_cache

import torch
from torch import nn

from turbo_physai.optimizations.models.openwam import enabled


@lru_cache(maxsize=1)
def lightop_ops():
    from lightop import op

    for name in ("rmsnorm_forward_autograd", "layernorm_forward_autograd"):
        if not hasattr(op, name):
            raise RuntimeError(f"Installed lightop does not provide {name}")
    return op


def _supported(x, *parameters):
    return (
        x.is_cuda
        and bool(getattr(torch.version, "hip", None))
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.shape[-1] >= 64
        and all(
            p is None or (p.dtype == x.dtype and p.is_contiguous()) for p in parameters
        )
    )


def lightop_rms_norm(x, weight, eps):
    if torch.compiler.is_compiling():
        return None
    if not _supported(x, weight):
        return None
    flat = x.reshape(-1, x.shape[-1])
    return (
        lightop_ops()
        .rmsnorm_forward_autograd(flat, weight, eps, torch.is_grad_enabled())
        .reshape_as(x)
    )


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if (
            enabled("lightop_norm")
            and not torch.compiler.is_compiling()
            and tuple(self.normalized_shape) == (x.shape[-1],)
            and _supported(x, self.weight, self.bias)
        ):
            flat = x.reshape(-1, x.shape[-1])
            return (
                lightop_ops()
                .layernorm_forward_autograd(
                    flat, self.weight, self.bias, self.eps, torch.is_grad_enabled()
                )
                .reshape_as(x)
            )
        return super().forward(x)
