# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""TF32 QK multiplication for unmasked ACT attention on HCU.

The forward uses TF32 inputs with FP32 accumulation. Backward uses the
standard Torch BMM implementation and supports first-order gradients only.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_kernel(a_ptr, b_ptr, out_ptr,
               M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
               AS0: tl.constexpr, AS1: tl.constexpr, AS2: tl.constexpr,
               BS0: tl.constexpr, BS1: tl.constexpr, BS2: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    mi = (tile // tl.cdiv(N, BN)) * BM + tl.arange(0, BM)
    ni = (tile % tl.cdiv(N, BN)) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.float32)
    for k0 in range(tl.cdiv(K, BK)):
        kk = k0 * BK + ki
        a = tl.load(a_ptr + batch * AS0 + mi[:, None] * AS1 + kk[None, :] * AS2,
                    (mi[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + batch * BS0 + kk[:, None] * BS1 + ni[None, :] * BS2,
                    (kk[:, None] < K) & (ni[None, :] < N), other=0.0)
        accum = tl.dot(a, b, accum, input_precision="tf32")
    tl.store(out_ptr + batch * M * N + mi[:, None] * N + ni[None, :], accum,
             (mi[:, None] < M) & (ni[None, :] < N))


class _QKTF32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key):
        if (query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0]
                or query.shape[2] != key.shape[1]):
            raise ValueError("QK tensors have incompatible batch or matrix dimensions")
        if not query.is_cuda or not key.is_cuda or query.device != key.device:
            raise ValueError("QK tensors must be on the same HCU device")
        if query.dtype != torch.float32 or key.dtype != torch.float32:
            raise ValueError("QK TF32 expects float32 inputs")
        if any(dim == 0 for dim in (*query.shape, key.shape[2])):
            raise ValueError("QK TF32 does not support empty dimensions")

        m, n, k = query.shape[1], key.shape[2], query.shape[2]
        output = torch.empty((query.shape[0], m, n), device=query.device, dtype=query.dtype)
        grid = (triton.cdiv(m, 64) * triton.cdiv(n, 128), query.shape[0])
        _qk_kernel[grid](query, key, output, m, n, k, *query.stride(), *key.stride(),
                         64, 128, 32, num_warps=4, num_stages=2)
        ctx.save_for_backward(query, key)
        return output

    @staticmethod
    def backward(ctx, gradient):
        if torch.is_grad_enabled():
            raise RuntimeError("QK TF32 supports first-order gradients only")
        query, key = ctx.saved_tensors
        return (torch.bmm(gradient, key.transpose(-2, -1)),
                torch.bmm(query.transpose(-2, -1), gradient))


def qk_bmm(query, key):
    """Compute ACT QK scores with TF32 multiplication and FP32 accumulation."""
    return _QKTF32.apply(query, key)
