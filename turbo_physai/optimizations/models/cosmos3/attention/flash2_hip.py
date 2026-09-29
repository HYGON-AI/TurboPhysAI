# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""HIP FlashAttention v2 support ported from the adapted Cosmos3 source.

Replaces ``flash2_attention`` (single from-import consumer:
``cosmos_framework.model.attention.frontend``), ``flash2_attention_check``
(varlen allowed on HIP), ``get_backend_list`` (flash2-first on HIP) and
``get_arch_tag`` (HIP fallback tag 90).  ``FLASH2_SUPPORTED`` is computed at
from __future__ import annotations

import time from the locked 2.7.x version range; on HIP the wheel is newer, so
the widened range is re-evaluated by ``recompute_flash2_supported`` below and
patched into the already-imported modules through the catalog declarations.
"""

import torch
from torch import Tensor

import cosmos_framework.model.attention.flash2 as _flash2_pkg
import cosmos_framework.model.attention.flash2.checks as _checks_mod
import cosmos_framework.model.attention.utils as _attn_utils
import cosmos_framework.model.attention.backends as _backends_mod
from cosmos_framework.model.attention.checks import assert_universal_tensor_checks
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.model.attention.utils.environment import is_torch_compiling
from cosmos_framework.model.attention.utils.version import version_in_range

from .._source_binding import bind_missing_globals

try:
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
except ImportError:  # flash2 stubs path; declarations depending on this fail cleanly
    flash_attn_func = None
    flash_attn_varlen_func = None

try:
    from flash_attn.flash_attn_interface import (
        _wrapped_flash_attn_varlen_backward,
        _wrapped_flash_attn_varlen_forward,
    )
except (AttributeError, ImportError):
    _wrapped_flash_attn_varlen_backward = None
    _wrapped_flash_attn_varlen_forward = None

# HCU uses a newer HIP FlashAttention wheel; keep the upstream CUDA range
# unchanged and widen validation only for HIP builds.
FLASH_ATTENTION_V2_MAX_VERSION = (
    "2.8.4.post1" if getattr(torch.version, "hip", None) else "2.7.4.post1"
)


def flash2_supported() -> bool:
    """Version-widened re-implementation of flash2.__init__.flash2_supported."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn
    except Exception:
        return False
    version_str = getattr(flash_attn, "__version__", None)
    if version_str is None:
        from importlib.metadata import version

        version_str = version("flash_attn")
    return version_in_range(
        version_str, _flash2_pkg.FLASH_ATTENTION_V2_MIN_VERSION, FLASH_ATTENTION_V2_MAX_VERSION
    )


FLASH2_SUPPORTED = flash2_supported()


class _CosmosFlashAttnVarlenFunc(torch.autograd.Function):
    """HIP varlen wrapper that avoids data-dependent cuseq trimming in FA."""

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        softcap: float,
        alibi_slopes: Tensor | None,
        deterministic: bool,
        return_softmax: bool,
    ):
        assert _wrapped_flash_attn_varlen_forward is not None
        assert _wrapped_flash_attn_varlen_backward is not None
        cu_seqlens_q = cu_seqlens_q.clamp(max=q.shape[0])
        cu_seqlens_k = cu_seqlens_k.clamp(max=k.shape[0])
        out, softmax_lse, s_dmask, rng_state = _wrapped_flash_attn_varlen_forward(
            q,
            k,
            v,
            None,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            alibi_slopes,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            window_size[0],
            window_size[1],
            softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.qk_headdim = q.shape[-1]
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, s_dmask)

    @staticmethod
    def backward(ctx, dout: Tensor, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        assert _wrapped_flash_attn_varlen_backward is not None
        _wrapped_flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : ctx.qk_headdim]
        dk = dk[..., : ctx.qk_headdim]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None


def _cosmos_flash_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    **kwargs,
):
    """Use the HCU direct wrapper when available, otherwise public FA API."""

    if (
        kwargs
        or _wrapped_flash_attn_varlen_forward is None
        or _wrapped_flash_attn_varlen_backward is None
    ):
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            **kwargs,
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    return _CosmosFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


def flash2_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    causal_type: CausalType | None = None,
    scale: float | None = None,
    cumulative_seqlen_Q: Tensor | None = None,
    cumulative_seqlen_KV: Tensor | None = None,
    max_seqlen_Q: int | None = None,
    max_seqlen_KV: int | None = None,
    return_lse: bool = False,
    backend_kwargs: dict | None = None,
    deterministic: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    Runs Flash Attention v2 on given operands (Q, K, V) with the heads-last contiguous layout
        (`[batch, seqlen, heads, head_dim]`).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with heads-last contiguous layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): whether or not causal masking is enabled. Default is False.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        scale (float | None): Dot product scale (attention scale). Defaults to head_dim ** -0.5.

        cumulative_seqlen_Q (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Tensor | None): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (int | None): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (int | None): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        return_lse (bool): Whether to return the logsumexp values. Default is False.

        backend_kwargs (dict | None): Key-value pair for passing arguments specific to Flash's
            attention operator, if any.

        deterministic (bool): Deterministic backward pass required.

    Returns:
        output (Tensor): 4-D output tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): logsumexp tensor, with the heads-last contiguous layout
            (`[batch, seqlen, heads, 1]`). Only returned when return_lse is True.
            NOTE: this tensor is not contiguous in this backend (Flash2) and it should not be made
            contiguous unless we can guarantee its results aren't merged via `merge_attentions`.
    """

    is_varlen = cumulative_seqlen_Q is not None
    assert_universal_tensor_checks(query, key, value)

    backend_kwargs = backend_kwargs.copy() if backend_kwargs is not None else {}
    # Determinism in backend_kwargs supersedes primary flag, if set to True
    if "deterministic" in backend_kwargs:
        deterministic = deterministic or backend_kwargs["deterministic"]
        del backend_kwargs["deterministic"]

    assert flash2_attention_check(
        query_shape=query.shape,
        key_shape=key.shape,
        value_shape=value.shape,
        dtype=query.dtype,
        device=query.device,
        requires_grad=query.requires_grad or key.requires_grad or value.requires_grad,
        is_causal=is_causal,
        causal_type=causal_type,
        is_varlen=is_varlen,
        deterministic=deterministic,
        raise_error=True,
    )

    # This check introduces recompiles
    if not is_torch_compiling():
        if is_varlen and max_seqlen_Q == max_seqlen_KV == 0:
            raise NotImplementedError(
                "You're trying to use varlen attention with the flash2 backend and "
                "an empty batch, which is not yet supported by flash2."
            )

    scale = scale if scale is not None else query.shape[-1] ** -0.5

    if is_varlen:
        assert query.shape[0] == key.shape[0] == value.shape[0] == 1
        q = query.squeeze(0)  # [total_tokens,H,D]
        k = key.squeeze(0)  # [total_tokens,Hkv,D]
        v = value.squeeze(0)  # [total_tokens,Hkv,Dv]
        assert q.dim() == k.dim() == v.dim() == 3
        varlen_func = (
            _cosmos_flash_attn_varlen_func
            if getattr(torch.version, "hip", None)
            else flash_attn_varlen_func
        )
        out, lse_, _ = varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cumulative_seqlen_Q,
            cu_seqlens_k=cumulative_seqlen_KV,
            max_seqlen_q=max_seqlen_Q,
            max_seqlen_k=max_seqlen_KV,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
            # block_table=None,
        )
        assert out.dim() == 3  # [total_tokens,H,Dv]
        assert lse_.dim() == 2  # [H,total_tokens]

        output = out.unsqueeze(0)  # [1,total_tokens,H,Dv]
        lse = lse_.unsqueeze(0)  # [1,H,total_tokens]

    else:
        output, lse, _ = flash_attn_func(  # output: [B,N,H,Dv], lse: [B,H,N]
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            return_attn_probs=True,
            deterministic=deterministic,
            **backend_kwargs,
            # window_size=(-1, -1),
            # dropout_p=0.0,
            # softcap=0.0, # 0.0 means deactivated
            # alibi_slopes=None,
        )

    assert isinstance(output, Tensor)
    assert isinstance(lse, Tensor)
    assert output.dim() == 4  # [B,N,H,Dv] or [1,total_tokens,H,Dv]
    assert lse.dim() == 3  # [B,H,N] or [1,H,total_tokens]

    # NOTE: Do NOT call .contiguous on LSE, otherwise Attention Merging backward pass will be
    # incorrect. All output and lse tensors passed into `merge_attentions` must have the same data
    # pointer as their corresponding attention autograd ops!
    lse = lse.permute(0, 2, 1)  # [B,N,H] or [1,total_tokens,H]

    if return_lse:
        return output, lse

    return output


def flash2_attention_check(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    is_causal: bool,
    causal_type: CausalType,
    is_varlen: bool,
    deterministic: bool = False,
    raise_error: bool = False,
) -> bool:
    """
    Input validation function for the flash2 backend.

    Parameters:
        query_shape (torch.Size): Shape of 4-D query tensor (`[batch, seqlen, heads, head_dim]`).

        key_shape (torch.Size): Shape of 4-D key tensor (`[batch, seqlen_kv, heads_kv, head_dim]`).

        value_shape (torch.Size): Shape of 4-D value tensor (`[batch, seqlen_kv, heads_kv, head_dim_v]`).

        dtype (torch.dtype): Data type of tensors.

        device (torch.device): Device of tensors.

        requires_grad (bool): Whether tensors require gradients (training vs inference).

        is_causal (bool): whether or not causal masking is enabled.

        causal_type (CausalType): causal masking mode. Choices: `CausalType.TopLeft`,
            `CausalType.BottomRight`. Required when `is_causal = True`.

        is_varlen (bool): whether or not a variable length (varlen) use case. Must be inferred
            beforehand based on arguments such as seqlens_{Q,KV} or cumulative_seqlen_{Q,KV} being
            passed.

        deterministic (bool): Deterministic backward pass required.

        raise_error (bool): whether to raise an error if any checks fail or no backend is selected,
            instead of just returning False. Default is False.

    Returns:
        success (bool): whether use case is compatible with flash2 backend.

    """
    target_fn = partial(log_or_raise_error, raise_error=raise_error)

    if not FLASH2_SUPPORTED:
        target_fn(
            "Flash Attention v2 (flash2) is not supported in this environment. Run with debug logs to find out why, or choose another backend.",
            exception=RuntimeError,
        )
        return False

    if is_varlen and getattr(torch.version, "hip", None) is None:
        target_fn(
            "Flash Attention v2 (flash2) varlen is banned due to instability. Please choose another backend.",
            exception=ValueError,
        )
        return False

    arch_tag = get_arch_tag(device)
    fwd_dtypes = get_fwd_dtypes(arch_tag)
    bwd_dtypes = get_bwd_dtypes(arch_tag)
    if not attention_tensor_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        dtype=dtype,
        requires_grad=requires_grad,
        supported_dtypes_forward=fwd_dtypes,
        supported_dtypes_backward=bwd_dtypes,
        supports_mla=False,
        supports_gqa_mqa=True,
        raise_error=raise_error,
        backend_name="Flash Attention v2 (flash2)",
    ):
        target_fn("Flash Attention v2 (flash2) does not support the given inputs.", exception=RuntimeError)
        return False

    # Verifies causal_type is a CausalType instance when is_causal
    # Verifies DontCare is not used unless seqlen_q == seqlen_kv
    attention_param_checks(
        query_shape=query_shape,
        key_shape=key_shape,
        value_shape=value_shape,
        is_causal=is_causal,
        causal_type=causal_type,
    )

    if is_causal and causal_type not in [CausalType.BottomRight, CausalType.DontCare]:
        target_fn("Flash Attention v2 only supports bottom-right causal masking.", exception=RuntimeError)
        return False

    return True


def get_backend_list(arch_tag: int) -> list[str]:
    """
    Returns list of supported backends according to arch tag (attention.utils.get_arch_tag).
    Backends are ordered based on their known performance levels, so that the best-performing
    compatible backend is selected.

    The returned list can be filtered via environment variable.
    See `filter_attention_backends` for details.

    Parameters:
        arch_tag (int): Arch tag for the current CUDA device. Example: 80 for A100, 90 for H100.

    Returns:
        backend_list (list[str]): a list of backend names (string). Empty if device is not supported.

    """

    if arch_tag < 75:
        log.debug(f"Minimum architecture supported for Attention is 75, got {arch_tag=}.")
        return []

    default_backends = []
    if getattr(torch.version, "hip", None):
        default_backends = ["flash2"]
    elif arch_tag == 90:
        default_backends = [
            "flash3",
            "cudnn",
            "natten",
            "flash2",
        ]
    elif arch_tag in [100, 103]:
        default_backends = [
            "cudnn",
            "natten",
            "flash2",
        ]
    elif arch_tag in [110, 120, 121]:
        default_backends = [
            "cudnn",
            "flash2",
            "natten",
        ]
    elif arch_tag >= 80:
        default_backends = [
            "flash2",
            "cudnn",
            "natten",
        ]
    else:
        default_backends = ["natten"]

    # Apply environment variable filtering
    return filter_attention_backends(default_backends)


def get_arch_tag(device: torch.device | None = None) -> int:
    """
    Returns the compute capability of a CUDA/HIP device, otherwise returns 0.

    HIP-backed HCU builds expose the torch.cuda API but may not provide CUDA
    version metadata or a queryable compute capability.  Use the queried value
    when available and fall back to the HCU backend's Hopper-compatible tag.
    """
    is_cuda_like = torch.cuda.is_available() and (
        torch.version.cuda or getattr(torch.version, "hip", None)
    )
    if is_cuda_like and (device is None or device.type == "cuda"):
        try:
            major, minor = torch.cuda.get_device_capability(device)
            arch_tag = major * 10 + minor
            if arch_tag > 0:
                return arch_tag
        except Exception:
            pass
        if getattr(torch.version, "hip", None):
            return 90
    return 0


# flash2_attention_check reads the import-time FLASH2_SUPPORTED flag; rebind it
# to the widened evaluation above so a HIP-only wheel passes the check.
bind_missing_globals(globals(), _checks_mod, _backends_mod, _flash2_pkg, _attn_utils)
