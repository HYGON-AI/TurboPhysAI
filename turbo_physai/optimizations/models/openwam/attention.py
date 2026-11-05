# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

"""Exact mask-preserving fast paths for padding cross-attention."""

import logging
import weakref
from collections import OrderedDict
from functools import lru_cache

import torch

from turbo_physai.optimizations.models.openwam.settings import enabled

logger = logging.getLogger(__name__)
_PADDING_PLANS = OrderedDict()


@lru_cache(maxsize=1)
def _varlen_kernel():
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        logger.warning(
            "[optimizations] FA2 padding skipped: flash_attn_varlen_func unavailable"
        )
        return None
    from turbo_physai.optimizations.models.openwam._fa2_varlen import (
        packed_varlen_kernel,
    )

    return packed_varlen_kernel(flash_attn_varlen_func)


def _padding_keys(mask, batch, query_len, key_len):
    # A stride-zero query axis proves query-independent masking without a GPU reduction.
    # An arbitrary dense mask, even if its rows happen to match, stays on SDPA.
    if (
        mask.ndim not in (3, 4)
        or mask.dtype != torch.bool
        or mask.shape[0] != batch
        or mask.shape[-1] != key_len
    ):
        return None
    if mask.ndim == 4 and mask.shape[1] == 1:
        if mask.shape[2] in (1, query_len) and (
            mask.shape[2] == 1 or mask.stride(2) == 0
        ):
            return mask[:, 0, 0, :]
    if mask.ndim == 3:
        if mask.shape[1] in (1, query_len) and (
            mask.shape[1] == 1 or mask.stride(1) == 0
        ):
            return mask[:, 0, :]
    return None


def _padding_signature(mask, keep):
    root = mask
    while root._base is not None:
        root = root._base
    try:
        signature = (
            keep.data_ptr(),
            tuple(keep.shape),
            tuple(keep.stride()),
            keep.device,
            root._version,
        )
    except RuntimeError:  # Inference tensors have no version counter.
        signature = None
    return root, signature


def _remember_padding(root, signature, plan):
    if signature is not None:
        _PADDING_PLANS[signature] = weakref.ref(root), plan
    while len(_PADDING_PLANS) > 16:
        _PADDING_PLANS.popitem(last=False)


def _padding_plan(mask, keep):
    root, signature = _padding_signature(mask, keep)
    cached = _PADDING_PLANS.get(signature)
    if cached is not None and cached[0]() is root:
        _PADDING_PLANS.move_to_end(signature)
        return cached[1]
    plan = _build_padding_plan(keep)
    _remember_padding(root, signature, plan)
    return plan


def _build_padding_plan(keep):
    lengths = keep.sum(dim=-1, dtype=torch.int32)
    nonempty = lengths > 0
    if keep.device.type == "cpu":
        nonempty._openwam_all_true = bool(nonempty.all())
    # FA2 builds differ in support for zero-length K. Give empty rows a dummy
    # first key, then zero their result (and gradients), matching SDPA semantics.
    positions = torch.arange(keep.shape[1], device=keep.device)
    packed_keep = keep | ((~nonempty).unsqueeze(1) & (positions == 0).unsqueeze(0))
    indices = packed_keep.reshape(-1).nonzero(as_tuple=False).flatten()
    offsets = torch.cat(
        [lengths.new_zeros(1), lengths.clamp_min(1).cumsum(0, dtype=torch.int32)]
    )
    return indices, offsets, nonempty


def cpu_context_mask(inputs, *, append_proprio=False, sample_mask=None):
    """Build a final mask and prime packing from CPU metadata, or fall back.

    Called after text trimming and before proprio append. Never pulls a GPU
    mask to the host. The cache uses the final device tensor's identity/version;
    arbitrary masks and subsequent mutations keep the original packing path.
    """
    if not (enabled("padding_plan_cpu") and enabled("fa2_padding")):
        return None
    context = inputs.get("context")
    if not isinstance(context, torch.Tensor) or context.ndim != 3:
        return None
    batch, length = context.shape[:2]
    if not batch or not length:
        return None
    keep = inputs.get("context_mask")
    if keep is None:
        lengths = inputs.get("seq_lens")
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.device.type != "cpu"
            or lengths.shape != (batch,)
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            return None
        keep = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
    elif (
        not isinstance(keep, torch.Tensor)
        or keep.device.type != "cpu"
        or keep.dtype != torch.bool
        or keep.shape != (batch, length)
    ):
        return None
    if append_proprio:
        if sample_mask is None:
            extra = torch.ones((batch, 1), dtype=torch.bool)
        else:
            if (
                not isinstance(sample_mask, torch.Tensor)
                or sample_mask.device.type != "cpu"
            ):
                return None
            extra = sample_mask.to(dtype=torch.bool)
            if extra.ndim == 3:
                extra = extra.any(dim=-1)
            elif extra.ndim == 1:
                extra = extra.unsqueeze(-1)
            if extra.shape != (batch, 1):
                return None
        keep = torch.cat([keep, extra], dim=1)
    from turbo_physai.optimizations.models.openwam.host_async import (
        active as host_async_active,
    )
    from turbo_physai.optimizations.models.openwam.host_async import upload

    raw_plan = _build_padding_plan(keep)
    if host_async_active() and context.device.type == "cuda":
        # Pageable uploads wait for all queued work, stalling the host at the
        # start of the forward. Pinned copies are fresh, so no alias remains.
        plan = tuple(upload(value, context.device) for value in raw_plan)
        mask = upload(keep, context.device).contiguous()
    else:
        plan = tuple(value.to(context.device) for value in raw_plan)
        # Own storage even for CPU execution; no alias to user-mutable CPU masks.
        mask = keep.to(device=context.device, copy=True).contiguous()
    if hasattr(raw_plan[2], "_openwam_all_true"):
        plan[2]._openwam_all_true = raw_plan[2]._openwam_all_true
    root, signature = _padding_signature(mask, mask)
    _remember_padding(root, signature, plan)
    return mask


def _select_nonempty(out, nonempty):
    return torch.where(nonempty.view(-1, 1, 1), out, 0)


@lru_cache(maxsize=1)
def _compiled_nonempty():
    # Compile both forward and autograd backward instead of eager masked_fill's
    # clone/copy path. Selection preserves zero output for masked NaNs too.
    return torch.compile(_select_nonempty, dynamic=True, fullgraph=True)


def _mask_nonempty(out, nonempty):
    from turbo_physai.optimizations.models.openwam.sac import extras_enabled

    if getattr(nonempty, "_openwam_all_true", False) and not extras_enabled():
        return out
    if enabled("fa2_mask_fused"):
        return _compiled_nonempty()(out, nonempty)
    return out.masked_fill((~nonempty).view(-1, 1, 1), 0)


def fa2_padding(q, k, v, mask, num_heads):
    if (
        enabled("fa2_padding")
        and torch.compiler.is_compiling()
    ):
        return _eager_fa2_padding(q, k, v, mask, num_heads)
    return _fa2_padding(q, k, v, mask, num_heads)


def _fa2_padding(q, k, v, mask, num_heads):
    """Return (B,Sq,H*D) or None when the mask/backend cannot use this fast path."""
    if not enabled("fa2_padding") or mask is None:
        return None
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    if (
        k.dtype != q.dtype
        or v.dtype != q.dtype
        or k.device != q.device
        or v.device != q.device
    ):
        return None
    batch, sq, dim = q.shape
    sk = k.shape[1]
    if k.shape != v.shape or k.shape[0] != batch or k.shape[2] != dim:
        return None
    if (
        dim % num_heads
        or dim // num_heads > 256
        or dim // num_heads % 8
        or not sq
        or not sk
    ):
        return None
    keep = _padding_keys(mask, batch, sq, sk)
    if keep is None or mask.device != q.device:
        return None
    kernel = _varlen_kernel()
    if kernel is None:
        return None
    indices, cu_k, nonempty = _padding_plan(mask, keep)
    head_dim = dim // num_heads
    qp = q.reshape(batch * sq, num_heads, head_dim).contiguous()
    kp = (
        k.reshape(batch * sk, num_heads, head_dim).index_select(0, indices).contiguous()
    )
    vp = (
        v.reshape(batch * sk, num_heads, head_dim).index_select(0, indices).contiguous()
    )
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sq
    out = kernel(qp, kp, vp, cu_q, cu_k, sq, sk, dropout_p=0.0, causal=False)
    return _mask_nonempty(out.reshape(batch, sq, dim), nonempty)


_eager_fa2_padding = torch.compiler.disable(_fa2_padding)
