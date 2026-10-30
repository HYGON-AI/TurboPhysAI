# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
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
    if enabled("global_compile") and torch.compiler.is_compiling():
        return None
    from turbo_physai.optimizations.models.openwam.sac import (
        extras_enabled,
        native_mot_expression,
    )

    if (
        extras_enabled()
        and not torch.compiler.is_compiling()
        and native_mot_expression()
    ):
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
        native_sac = False
        from turbo_physai.optimizations.models.openwam.sac import (
            extras_enabled,
            native_mot_expression,
        )

        if extras_enabled() and not torch.compiler.is_compiling():
            native_sac = native_mot_expression()
        if (
            enabled("lightop_norm")
            and not native_sac
            and not (enabled("global_compile") and torch.compiler.is_compiling())
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
