# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Muon 的元素级融合 kernel：一组参数一次 launch，复用一张指针表。

单个 elementwise op 在 HCU 上跑不满带宽，而 update/apply 阶段每个参数只有两次读写、张量数量却上千。
这里把所有地址收进一张 int64 指针表，用一个 kernel 做完整组；
表本身按地址元组缓存、用 pinned 内存异步拷贝，下一个 step 直接命中。
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

#: 纯流式 elementwise；4096 fp32 lane/程序，队列够深的同时不溢出。
_BLOCK = 4096

#: 指针表缓存
_PTR_CACHE: Dict[tuple, tuple] = {}
_PTR_CACHE_MAX = 4096

if _HAS_TRITON:

    @triton.jit
    def _muon_momentum_kernel(
        grad_ptr,
        buf_ptr,
        out_ptr,
        momentum,
        n_elements,
        NESTEROV: tl.constexpr,
        OUT_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
        b = tl.load(buf_ptr + offs, mask=mask, other=0.0)

        # buf_local.lerp_(grad_local, 1 - momentum) == b + (1-m)*(g - b)
        b_new = b + (1.0 - momentum) * (g - b)
        tl.store(buf_ptr + offs, b_new, mask=mask)

        # grad_local.lerp(buf_local, momentum) == g + m*(b_new - g)
        if NESTEROV:
            u = g + momentum * (b_new - g)
        else:
            u = b_new
        if OUT_BF16:
            tl.store(out_ptr + offs, u.to(tl.bfloat16), mask=mask)
        else:
            tl.store(out_ptr + offs, u, mask=mask)

    @triton.jit
    def _muon_apply_kernel(
        p_ptr,
        ortho_ptr,
        decay,
        lr_step,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        o = tl.load(ortho_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        # p_local.mul_(decay) 后接 p_local.add_(ortho, alpha=-lr_step)
        p = p * decay
        p = p - lr_step * o
        tl.store(p_ptr + offs, p, mask=mask)

    @triton.jit
    def _mt_apply_kernel(
        p_ptrs,
        o_ptrs,
        decay,
        lr_step,
        n_elements,
        ORTHO_BF16: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """整组同 shape 参数只进行一次 launch。grid = (N, blocks)"""
        tid = tl.program_id(0)
        bid = tl.program_id(1)
        p_ptr = tl.load(p_ptrs + tid).to(tl.pointer_type(tl.float32))
        if ORTHO_BF16:
            o_ptr = tl.load(o_ptrs + tid).to(tl.pointer_type(tl.bfloat16))
        else:
            o_ptr = tl.load(o_ptrs + tid).to(tl.pointer_type(tl.float32))

        offs = bid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        o = tl.load(o_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        p = p * decay
        p = p - lr_step * o
        tl.store(p_ptr + offs, p, mask=mask)

    @triton.jit
    def _pack_kernel(
        X, Y, M: tl.constexpr, K: tl.constexpr, B: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """packed gather 布局 [B,M,K] stride (M,1,B*M) -> 连续 BF16。"""
        b = tl.program_id(2).to(tl.int64)
        m = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        k = tl.program_id(1).to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        mask = (m[:, None] < M) & (k[None, :] < K)
        value = tl.load(X + k[None, :] * B * M + b * M + m[:, None], mask=mask, other=0)
        tl.store(Y + b * M * K + m[:, None] * K + k[None, :], value, mask=mask)
else:
    _muon_momentum_kernel = None
    _muon_apply_kernel = None
    _mt_apply_kernel = None
    _pack_kernel = None

# momentum 融合 kernel 的最小元素数门槛。小张量的 ATen lerp 已经接近带宽上限，融合反而更慢；只有 200 万元素以上才有收益，所以小的直接使用 ATen。
_FUSE_MIN_ELEMS = 2 << 20

def _can_fuse_apply(param: Tensor, ortho: Tensor) -> bool:
    """fp32 参数 + fp32/bf16 ortho，同元素数、同连续布局才融合。"""
    return (
        _HAS_TRITON
        and param.is_cuda
        and param.dtype == torch.float32
        and ortho.dtype in (torch.float32, torch.bfloat16)
        and param.is_contiguous()
        and ortho.is_contiguous()
        and param.numel() == ortho.numel()
    )

def can_fuse(grad: Tensor, buf: Tensor, out: Tensor) -> bool:
    """融合 kernel 只接受连续、同 device 的 fp32 grad/buffer。"""
    if not _HAS_TRITON:
        return False
    if not grad.is_cuda or grad.dtype != torch.float32:
        return False
    if buf.dtype != torch.float32:
        return False
    if out.dtype not in (torch.float32, torch.bfloat16):
        return False
    if not (grad.is_contiguous() and buf.is_contiguous() and out.is_contiguous()):
        return False
    return grad.numel() == buf.numel() == out.numel()

def can_fuse_multi(tensors: Sequence[Tensor], dtype: torch.dtype) -> bool:
    """整组 tensor 是否同元素数、连续、指定 dtype、同 device。"""
    if not _HAS_TRITON or not tensors:
        return False
    first = tensors[0]
    if not first.is_cuda or first.dtype != dtype or not first.is_contiguous():
        return False
    n = first.numel()
    if n == 0:
        return False
    return all(
        t.dtype == dtype and t.is_contiguous() and t.numel() == n and t.device == first.device
        for t in tensors
    )

def momentum_nesterov_(
    grad: Tensor, buf: Tensor, out: Tensor, momentum: float, nesterov: bool
) -> None:
    """buf.lerp_(grad, 1-m) + update = grad.lerp(buf, m)，一次读写。"""
    n = out.numel()
    if n == 0:
        return
    _muon_momentum_kernel[(triton.cdiv(n, _BLOCK),)](
        grad, buf, out, float(momentum), n,
        NESTEROV=bool(nesterov),
        OUT_BF16=out.dtype == torch.bfloat16,
        BLOCK=_BLOCK,
        num_warps=8,
    )

def muon_apply_(param: Tensor, ortho: Tensor, decay: float, lr_step: float) -> None:
    """param.mul_(decay) + param.add_(ortho, alpha=-lr_step)，一步完成。"""
    n = param.numel()
    if n == 0:
        return
    _muon_apply_kernel[(triton.cdiv(n, _BLOCK),)](
        param, ortho, float(decay), float(lr_step), n,
        BLOCK=_BLOCK,
        num_warps=8,
    )

def muon_apply_multi_(params: Sequence[Tensor], orthos: Sequence[Tensor],
                      decay: float, lr_step: float) -> None:
    """整组只进行一次 launch"""
    if not params:
        return
    n = params[0].numel()
    device = params[0].device
    _mt_apply_kernel[(len(params), triton.cdiv(n, _BLOCK))](
        _ptr_array(params, device),
        _ptr_array(orthos, device),
        float(decay),
        float(lr_step),
        n,
        ORTHO_BF16=orthos[0].dtype == torch.bfloat16,
        BLOCK=_BLOCK,
        num_warps=8,
    )

def _ptr_array(tensors: Sequence[Tensor], device: torch.device) -> Tensor:
    """把一组 tensor 的地址打包成 int64 表，按地址元组缓存"""
    key = tuple(t.data_ptr() for t in tensors)
    cached = _PTR_CACHE.get(key)
    if cached is not None:
        return cached[0]

    n = len(key)
    if device.type != "cuda" or n == 0 or torch.cuda.is_current_stream_capturing():
        arr, pin = torch.tensor(key, dtype=torch.int64, device=device), None
    else:
        pin = torch.tensor(key, dtype=torch.int64, pin_memory=True)
        arr = torch.empty(n, dtype=torch.int64, device=device)
        arr.copy_(pin, non_blocking=True)

    if len(_PTR_CACHE) >= _PTR_CACHE_MAX:
        _PTR_CACHE.pop(next(iter(_PTR_CACHE)))
    _PTR_CACHE[key] = (arr, pin)
    return arr
