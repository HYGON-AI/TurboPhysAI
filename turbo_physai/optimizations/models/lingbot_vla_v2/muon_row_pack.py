# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""owner-NS 打包/解包的向量化 kernel。"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False

_BLOCK = 4096
_BLOCK_R = 32
_BLOCK_K = 32


if _HAS_TRITON:

    @triton.jit
    def _rows_flat_kernel(
        mat_ptr,
        flat_ptr,
        COUNT,
        NROW,
        K,
        WS: tl.constexpr,
        PACK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        nkb = tl.cdiv(K, BLOCK)
        kb = pid % nkb
        t = pid // nkb
        row = t % NROW
        m = t // NROW

        chunk = tl.cdiv(NROW, WS)
        s = row // chunk
        start = s * chunk
        length = tl.minimum(chunk, NROW - start)

        k = kb * BLOCK + tl.arange(0, BLOCK)
        kmask = k < K
        m64 = m.to(tl.int64)
        base_mat = (m64 * NROW + row) * K
        base_flat = (
            start.to(tl.int64) * COUNT * K
            + m64 * (length.to(tl.int64) * K)
            + (row - start).to(tl.int64) * K
        )
        if PACK:
            v = tl.load(mat_ptr + base_mat + k, mask=kmask)
            tl.store(flat_ptr + base_flat + k, v, mask=kmask)
        else:
            v = tl.load(flat_ptr + base_flat + k, mask=kmask)
            tl.store(mat_ptr + base_mat + k, v, mask=kmask)

    @triton.jit
    def _rows_flat_T_kernel(
        matT_ptr,
        flat_ptr,
        COUNT,
        NROW,
        K,
        WS: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        k = tl.program_id(2) * BLOCK_K + tl.arange(0, BLOCK_K)
        rmask = r < NROW
        kmask = k < K

        chunk = tl.cdiv(NROW, WS)
        s = r // chunk
        start = s * chunk
        length = tl.minimum(chunk, NROW - start)

        m64 = m.to(tl.int64)
        off_load = m64 * K * NROW + k[:, None].to(tl.int64) * NROW + r[None, :]
        blk = tl.load(
            matT_ptr + off_load,
            mask=kmask[:, None] & rmask[None, :],
            other=0,
        )
        blk = tl.trans(blk)

        off_store = (
            (start.to(tl.int64) * COUNT * K)[:, None]
            + (m64 * (length.to(tl.int64) * K))[:, None]
            + ((r - start).to(tl.int64) * K)[:, None]
            + k[None, :]
        )
        tl.store(flat_ptr + off_store, blk, mask=rmask[:, None] & kmask[None, :])


def _check(mat: Tensor, flat: Tensor, count: int, world_size: int):
    if not _HAS_TRITON or world_size <= 1:
        return None
    if mat.ndim != 3 or mat.shape[0] != count:
        return None
    if not flat.is_contiguous() or flat.dtype != mat.dtype or flat.device != mat.device:
        return None
    if not mat.is_cuda:
        return None
    nrow, k = int(mat.shape[1]), int(mat.shape[2])
    if nrow < world_size or k <= 0 or count <= 0:
        return None
    if flat.numel() != count * nrow * k:
        return None
    if mat.is_contiguous():
        return (count, nrow, k, False)
    if mat.mT.is_contiguous() and mat.stride() == (nrow * k, 1, nrow):
        return (count, nrow, k, True)
    return None


def _row_block(k: int) -> int:
    return min(_BLOCK, max(256, triton.next_power_of_2(k)))


def pack_rows(mat: Tensor, flat: Tensor, *, count: int, world_size: int,
              block: int = None) -> bool:
    params = _check(mat, flat, count, world_size)
    if params is None:
        return False
    count_, nrow, k, transposed = params
    if transposed:
        grid = (count_, triton.cdiv(nrow, _BLOCK_R), triton.cdiv(k, _BLOCK_K))
        _rows_flat_T_kernel[grid](
            mat.mT, flat, count_, nrow, k,
            WS=world_size, BLOCK_R=_BLOCK_R, BLOCK_K=_BLOCK_K, num_warps=4,
        )
        return True
    block = _row_block(k) if block is None else block
    grid = (count_ * nrow * triton.cdiv(k, block),)
    _rows_flat_kernel[grid](
        mat, flat, count_, nrow, k,
        WS=world_size, PACK=True, BLOCK=block, num_warps=4,
    )
    return True


def unpack_rows(flat: Tensor, mat: Tensor, *, count: int, world_size: int,
                block: int = None) -> bool:
    params = _check(mat, flat, count, world_size)
    if params is None:
        return False
    count_, nrow, k, transposed = params
    if transposed:
        return False
    block = _row_block(k) if block is None else block
    grid = (count_ * nrow * triton.cdiv(k, block),)
    _rows_flat_kernel[grid](
        mat, flat, count_, nrow, k,
        WS=world_size, PACK=False, BLOCK=block, num_warps=4,
    )
    return True
