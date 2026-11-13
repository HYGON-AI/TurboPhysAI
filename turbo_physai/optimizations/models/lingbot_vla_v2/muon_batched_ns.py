# Copyright 2026 Robbyant Team and/or its affiliates
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""Muon 的 Newton-Schulz：把独立的小矩阵打包成一个 batch 一起算。

NS 的每一步都是 ``[M,M] @ [M,K]`` 的小矩阵乘，单个跑不满计算单元。且尾部矩阵之间互不影响，所以可以 stack 成 [B,M,K] 用 baddbmm 一次计算；
打包前按 A=[B,M,M] 的 bf16 字节数控制 batch 大小，避免峰值显存暴增。
packed gather 布局还可以使用融合 kernel 直接产出连续 bf16。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from lingbotvla.optim.muon import DEFAULT_NS_COEFFICIENTS, DEFAULT_NS_STEPS, EPS

from .muon_fused_kernels import _FUSE_MIN_ELEMS, _HAS_TRITON, _pack_kernel, triton

def can_pack(x: Tensor, compute_dtype: torch.dtype) -> bool:
    """是否是 packed gather 布局、且值得用融合 cast kernel 重排。"""
    if not _HAS_TRITON or not x.is_cuda or torch.version.hip is None:
        return False
    if x.dtype != torch.float32 or compute_dtype != torch.bfloat16 or x.ndim != 3:
        return False
    batch, rows, cols = x.shape
    return (
        batch >= 4
        and rows <= cols
        and x.numel() >= _FUSE_MIN_ELEMS
        and x.stride() == (rows, 1, batch * rows)
    )

def _tile_config(shape: tuple) -> tuple:
    """按 shape 选 (BLOCK_M, BLOCK_K, warps)。"""
    if shape == (32, 2560, 4096):
        return 128, 32, 4
    if shape == (8, 2560, 9728):
        return 64, 128, 8
    # 兜底，含 (32, 2560, 9728)
    return 64, 32, 4

def pack_for_ns(x: Tensor, compute_dtype: torch.dtype) -> Tensor:
    """支持时用融合 kernel 直接产出连续 BF16，否则退回 ``x.to``。"""
    if not can_pack(x, compute_dtype):
        return x.to(compute_dtype)
    batch, rows, cols = x.shape
    block_m, block_k, warps = _tile_config(tuple(x.shape))
    output = torch.empty(x.shape, device=x.device, dtype=compute_dtype)
    _pack_kernel[(triton.cdiv(rows, block_m), triton.cdiv(cols, block_k), batch)](
        x, output, rows, cols, batch, block_m, block_k, num_warps=warps,
    )
    return output

# NS 的 megabatch 切块：小矩阵在大 batch 下更快，大矩阵已饱和，所以按中间张量 A=[B,M,M] 的 bf16 字节数动态决定
_MEGABATCH_MIN_GROUP_SIZE = 32
_MEGABATCH_MAX_GROUP_SIZE = 128
_MEGABATCH_BYTES_BUDGET = 256 << 20


def _megabatch_chunk_size(rows: int, cols: int) -> int:
    """按 A=[B,M,M] 的 bf16 字节数定 batch大小：小矩阵放宽到 128，大矩阵保持 32。"""
    m = min(rows, cols)
    per_matrix = max(1, m * m * 2)  # bf16
    budget = _MEGABATCH_BYTES_BUDGET // per_matrix
    return max(_MEGABATCH_MIN_GROUP_SIZE, min(_MEGABATCH_MAX_GROUP_SIZE, budget))

@torch.no_grad()
def batched_newton_schulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
    compute_dtype: torch.dtype = torch.bfloat16,
    *,
    layout_pack: bool = False,
    local_slice: Optional[Tuple[int, int, int]] = None,
) -> Tensor:
    """对每个尾部 [M, K] 矩阵进行 quintic Newton-Schulz。"""
    return _iterate_newton_schulz(
        _prepare_newton_schulz(
            grad, ns_coefficients, ns_steps, compute_dtype, layout_pack=layout_pack,
        ),
        ns_coefficients, ns_steps, eps, local_slice=local_slice,
    )

def _prepare_newton_schulz(
    grad: Tensor,
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    compute_dtype: torch.dtype,
    *,
    layout_pack: bool,
) -> Tuple[Tensor, torch.dtype, bool]:
    """只返回计算存储和标量、不返回输入视图：调用方丢掉 gather 后就能释放 FP32。"""
    original_dtype = grad.dtype
    transposed = grad.size(-2) > grad.size(-1)
    # Pack after logical transpose, so the NS matrix always has M <= K.
    logical = grad.mT if transposed else grad
    ortho = pack_for_ns(logical, compute_dtype) if layout_pack else logical.to(compute_dtype)
    return ortho, original_dtype, transposed

def _iterate_newton_schulz(
    prepared: Tuple[Tensor, torch.dtype, bool],
    ns_coefficients: Tuple[float, float, float],
    ns_steps: int,
    eps: float,
    *,
    local_slice: Optional[Tuple[int, int, int]] = None,
) -> Tensor:
    """NS 迭代与输出转换，不依赖 gather 输入。"""
    ortho, original_dtype, transposed = prepared
    del prepared  # 别让这个 tuple 把 pre-normalization 张量留住
    a, b, c = ns_coefficients
    norm = ortho.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    ortho = ortho / norm
    del norm

    for _ in range(ns_steps):
        A = ortho @ ortho.mT
        if A.ndim == 2:
            gram_update = torch.addmm(A, A, A, beta=b, alpha=c)
            ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
        else:
            *batch, M_, K_ = ortho.shape
            B_ = 1
            for d in batch:
                B_ *= d
            A_3d = A.reshape(B_, M_, M_)
            ortho_3d = ortho.reshape(B_, M_, K_)
            gram_update = torch.baddbmm(A_3d, A_3d, A_3d, beta=b, alpha=c)
            ortho = torch.baddbmm(ortho_3d, gram_update, ortho_3d, beta=a).reshape(*batch, M_, K_)
            del A_3d, ortho_3d
        del gram_update
        del A

    if transposed:
        ortho = ortho.mT

    if local_slice is not None:
        # 只 cast 本地分片；显式 contiguous，apply 融合要求连续布局。
        dim, start, length = local_slice
        return ortho.narrow(dim, start, length).to(original_dtype).contiguous()
    return ortho.to(original_dtype)
