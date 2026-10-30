# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Compile only stateless tensor expressions, outside ZeRO/checkpoint orchestration."""

from functools import lru_cache


import torch
import torch.nn.functional as F

from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.sac import dispatcher


@lru_cache(maxsize=None)
def _compiled(fn):
    return torch.compile(fn, dynamic=True, fullgraph=True)


@dispatcher
def _run(fn, *args):
    from turbo_physai.optimizations.models.openwam.sac import (
        extras_enabled,
        run_pointwise,
    )

    if extras_enabled():
        return run_pointwise(fn, *args)
    if torch.compiler.is_compiling() and enabled("global_compile"):
        # The enclosing block already compiles this expression. A nested
        # dispatch would specialize this shared helper for every expression.
        return fn(*args)
    if args[0].is_cuda and enabled("pointwise_compile"):
        return _compiled(fn)(*args)
    return fn(*args)


def _rms(x, weight, eps):
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed.to(x.dtype) * weight


def rms_norm(x, weight, eps):
    if enabled("lightop_norm"):
        from turbo_physai.optimizations.models.openwam.norms import lightop_rms_norm

        result = lightop_rms_norm(x, weight, eps)
        if result is not None:
            return result
    return _run(_rms, x, weight, eps)


def _vae_norm_value(x, gamma, bias, dim, scale):
    return F.normalize(x, dim=dim) * scale * gamma + bias


def _vae_norm_silu_value(x, gamma, bias, dim, scale):
    return F.silu(_vae_norm_value(x, gamma, bias, dim, scale))


@lru_cache(maxsize=None)
def _compiled_vae_norm(dim, scale, rank, apply_silu):
    normalize = _vae_norm_silu_value if apply_silu else _vae_norm_value

    import torch._inductor

    options = (
        {"emulate_precision_casts": True}
        if "emulate_precision_casts" in torch._inductor.list_options()
        else {}
    )
    return torch.compile(normalize, dynamic=True, fullgraph=True, options=options)


def vae_normalize(x, gamma, bias, dim, scale, apply_silu=False):
    if torch.compiler.is_compiling() and enabled("global_compile"):
        normalize = _vae_norm_silu_value if apply_silu else _vae_norm_value
        return normalize(x, gamma, bias, dim, scale)
    if x.is_cuda and enabled("vae_pointwise_compile"):
        from torch._dynamo import mark_static

        if (
            x.ndim == 5
            and dim == 1
            and x.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            # Unit dimensions may carry different equivalent strides after
            # resampling. Use a metadata view to share the compiled graph.
            n, c, t, h, w = x.shape
            x = x.as_strided((n, c, t, h, w), (c * t * h * w, 1, h * w * c, w * c, c))
        # Keep the reduction width constant while sharing graphs across the
        # encoder's spatial/temporal shapes, without changing global limits.
        mark_static(x, dim)
        mark_static(gamma)
        if isinstance(bias, torch.Tensor):
            mark_static(bias)
        return _compiled_vae_norm(dim, scale, x.ndim, apply_silu)(
            x, gamma, bias, dim, scale
        )
    value = F.normalize(x, dim=dim) * scale * gamma + bias
    return F.silu(value) if apply_silu else value


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    return _run(_modulate, x, shift, scale)


def _gate(x, gate_value, residual):
    return x + gate_value * residual


def gate(x, gate_value, residual):
    return _run(_gate, x, gate_value, residual)


def _rotate(x, freqs):
    work_dtype = torch.float64
    pairs = x.to(work_dtype).reshape(*x.shape[:-1], -1, 2)
    real = freqs.real.to(device=x.device, dtype=work_dtype)
    imag = freqs.imag.to(device=x.device, dtype=work_dtype)
    left, right = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack(
        (left * real - right * imag, left * imag + right * real), dim=-1
    )
    return rotated.flatten(-2).to(x.dtype)


def rotate(x, freqs):
    return _run(_rotate, x, freqs)
