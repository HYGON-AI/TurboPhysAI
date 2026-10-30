# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Explicit video projections using the reference BW1000 bias2d addmm route.

Keep modules and Parameters in place. Hooks, custom forwards, compilation and
metadata outside the reference production case retain the original calls.
"""

from functools import lru_cache

import torch
from torch import nn

from turbo_physai.optimizations.models.openwam.settings import enabled
from turbo_physai.optimizations.models.openwam.sac import (
    dispatcher,
    projection_boundary,
    run_video_projection,
)

_HIP_TORCH_271 = (
    torch.version.hip is not None and torch.__version__.split("+")[0] == "2.7.1"
)
_CUBLAS_BACKEND = getattr(getattr(torch._C, "_BlasBackend", None), "Cublas", None)


@lru_cache(maxsize=16)
def _is_gfx936(device: torch.device) -> bool:
    try:
        arch = str(
            getattr(torch.cuda.get_device_properties(device), "gcnArchName", "UNKNOWN")
        ).split(":")[0]
        return arch == "gfx936"
    except Exception:
        return False


def _has_hooks(module: nn.Module) -> bool:
    return bool(
        getattr(module, "_forward_pre_hooks", None)
        or getattr(module, "_forward_hooks", None)
        or getattr(module, "_backward_pre_hooks", None)
        or getattr(module, "_backward_hooks", None)
    )


def _global_hooks() -> bool:
    return bool(
        getattr(nn.modules.module, "_global_forward_pre_hooks", None)
        or getattr(nn.modules.module, "_global_forward_hooks", None)
        or getattr(nn.modules.module, "_global_backward_pre_hooks", None)
        or getattr(nn.modules.module, "_global_backward_hooks", None)
    )


def _native(module, kind):
    return (
        type(module) is kind
        and getattr(module.forward, "__func__", None) is kind.forward
        and getattr(module, "_compiled_call_impl", None) is None
        and not _has_hooks(module)
    )


def can_use_bias2d(linear, x):
    # Return before backend/device queries in Dynamo. GLOBAL keeps its boundary.
    if (
        not enabled("linear_bias2d")
        or torch.compiler.is_compiling()
        or not _native(linear, nn.Linear)
        or _global_hooks()
        or not linear.training
        or not torch.is_grad_enabled()
    ):
        return False
    weight, bias = linear.weight, linear.bias
    if (
        type(x) is not torch.Tensor
        or type(weight) is not nn.Parameter
        or type(bias) is not nn.Parameter
        or not x.is_cuda
        or not weight.is_cuda
        or not bias.is_cuda
        or x.device != weight.device
        or x.device != bias.device
        or not _HIP_TORCH_271
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or bias.dtype != x.dtype
        or torch.is_autocast_enabled(x.device.type)
        or not x.requires_grad
        or not weight.requires_grad
        or not bias.requires_grad
        or tuple(x.shape) not in ((16, 360, 3072), (16, 360, 14336))
        or linear.in_features != x.shape[-1]
        or linear.out_features != 3072
        or tuple(weight.shape) != (3072, x.shape[-1])
        or tuple(bias.shape) != (3072,)
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not bias.is_contiguous()
    ):
        return False
    # Inspect the measured backend; never change the process-wide preference.
    if torch.backends.cuda.preferred_blas_library() != _CUBLAS_BACKEND:
        return False
    return _is_gfx936(x.device)


@dispatcher
def video_linear(linear, x):
    if projection_boundary():
        return run_video_projection(linear, x)
    return _video_linear(linear, x)


def _video_linear(linear, x):
    if not can_use_bias2d(linear, x):
        return linear(x)
    flat = x.reshape(-1, x.shape[-1])
    return torch.addmm(
        linear.bias.unsqueeze(0), flat, linear.weight.t(), beta=1, alpha=1
    ).reshape(
        *x.shape[:-1],
        linear.out_features,
    )


def cross_attention_kwargs(attention):
    if not enabled("linear_bias2d") or torch.compiler.is_compiling():
        return {}
    from openwam.model.video_backbone.wan.models.dit import CrossAttention

    if (
        not _native(attention, CrossAttention)
        or not attention.training
        or attention.has_image_input
        or _global_hooks()
    ):
        return {}
    return {"linear_bias2d": True}


def video_ffn(ffn, x):
    # Manual children must not bypass a Sequential hook or custom forward.
    if (
        not enabled("linear_bias2d")
        or torch.compiler.is_compiling()
        or not _native(ffn, nn.Sequential)
        or len(ffn) != 3
        or _global_hooks()
        or not _native(ffn[0], nn.Linear)
        or not _native(ffn[1], nn.GELU)
        or ffn[1].approximate != "tanh"
        or not _native(ffn[2], nn.Linear)
    ):
        return ffn(x)
    return video_linear(ffn[2], ffn[1](ffn[0](x)))
