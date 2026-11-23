# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""TF32 QK multiplication for ACT attention on HCU.

The forward uses TF32 inputs with FP32 accumulation. Backward uses the
standard Torch BMM implementation and supports first-order gradients only.
"""

import math

import torch
import torch.nn.functional as F
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


def _validate_qk(query, key):
    if (query.ndim != 3 or key.ndim != 3 or query.shape[0] != key.shape[0]
            or query.shape[2] != key.shape[1]):
        raise ValueError("QK tensors have incompatible batch or matrix dimensions")
    if not query.is_cuda or not key.is_cuda or query.device != key.device:
        raise ValueError("QK tensors must be on the same HCU device")
    if query.dtype != torch.float32 or key.dtype != torch.float32:
        raise ValueError("QK TF32 expects float32 inputs")
    if any(dim == 0 for dim in (*query.shape, key.shape[2])):
        raise ValueError("QK TF32 does not support empty dimensions")


@torch.library.custom_op("lerobot_act::qk_tf32", mutates_args=(), device_types="cuda")
def _qk_op(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    _validate_qk(query, key)
    m, n, k = query.shape[1], key.shape[2], query.shape[2]
    output = torch.empty((query.shape[0], m, n), device=query.device, dtype=query.dtype)
    grid = (triton.cdiv(m, 64) * triton.cdiv(n, 128), query.shape[0])
    _qk_kernel[grid](query, key, output, m, n, k, *query.stride(), *key.stride(),
                     64, 128, 32, num_warps=4, num_stages=2)
    return output


@_qk_op.register_fake
def _qk_fake(query, key):
    _validate_qk(query, key)
    return query.new_empty((query.shape[0], query.shape[1], key.shape[2]))


def _qk_setup_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs)


def _qk_backward(ctx, gradient):
    if torch.is_grad_enabled():
        raise RuntimeError("QK TF32 supports first-order gradients only")
    query, key = ctx.saved_tensors
    return (torch.bmm(gradient, key.transpose(-2, -1)),
            torch.bmm(query.transpose(-2, -1), gradient))


_qk_op.register_autograd(_qk_backward, setup_context=_qk_setup_context)


def qk_bmm(query, key):
    """Compute ACT QK scores with TF32 multiplication and FP32 accumulation."""
    return _qk_op(query, key)


def supports_attention_qk_tf32(query, key, value, embed_dim_to_check, num_heads,
                               in_proj_weight, in_proj_bias, bias_k, bias_v,
                               add_zero_attn, dropout_p, out_proj_weight,
                               out_proj_bias, training=True, key_padding_mask=None,
                               need_weights=True, attn_mask=None,
                               use_separate_proj_weight=False, q_proj_weight=None,
                               k_proj_weight=None, v_proj_weight=None, static_k=None,
                               static_v=None, average_attn_weights=True, is_causal=False):
    """Dispatch packed ACT attention, including key-padding masks."""
    return bool(
        query.is_cuda and query.dtype == torch.float32 and query.ndim == 3
        and key.ndim == 3 and value.ndim == 3 and need_weights
        and not is_causal and attn_mask is None
        and (key_padding_mask is None or (
            key_padding_mask.ndim == 2
            and key_padding_mask.shape == (query.shape[1], key.shape[0])
            and key_padding_mask.device == query.device
            and key_padding_mask.dtype in (torch.bool, torch.float32)))
        and not add_zero_attn and bias_k is None and bias_v is None
        and static_k is None and static_v is None and not use_separate_proj_weight
        and in_proj_weight is not None and key.shape == value.shape
        and query.shape[1] == key.shape[1]
        and query.shape[2] == key.shape[2] == embed_dim_to_check
        and num_heads > 0 and embed_dim_to_check % num_heads == 0
        and 0.0 <= dropout_p <= 1.0
    )


def attention_qk_tf32_wrapper(original, options):
    """Replace the packed ACT attention math path."""
    if options:
        raise ValueError("lerobot_act.qk_tf32 does not accept options")

    def wrapped(query, key, value, embed_dim_to_check, num_heads, in_proj_weight,
                in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p,
                out_proj_weight, out_proj_bias, training=True, key_padding_mask=None,
                need_weights=True, attn_mask=None, use_separate_proj_weight=False,
                q_proj_weight=None, k_proj_weight=None, v_proj_weight=None,
                static_k=None, static_v=None, average_attn_weights=True,
                is_causal=False):
        target_len, batch, embed_dim = query.shape
        source_len = key.shape[0]
        head_dim = embed_dim_to_check // num_heads
        q, k, v = F._in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
        q = q.view(target_len, batch * num_heads, head_dim).transpose(0, 1)
        k = k.view(source_len, batch * num_heads, head_dim).transpose(0, 1)
        v = v.view(source_len, batch * num_heads, head_dim).transpose(0, 1)
        q = q * math.sqrt(1.0 / float(head_dim))
        scores = qk_bmm(q, k.transpose(-2, -1))
        if key_padding_mask is not None:
            scores = scores.view(batch, num_heads, target_len, source_len)
            mask = key_padding_mask[:, None, None, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf'))
            else:
                scores = scores + mask
            scores = scores.view(batch * num_heads, target_len, source_len)
        probs = F.softmax(scores, dim=-1)
        if dropout_p > 0.0:
            probs = F.dropout(probs, p=dropout_p, training=training)
        output = torch.bmm(probs, v).transpose(0, 1).reshape(target_len * batch, embed_dim)
        output = F.linear(output, out_proj_weight, out_proj_bias)
        output = output.view(target_len, batch, output.size(1))
        weights = probs.view(batch, num_heads, target_len, source_len)
        if average_attn_weights:
            weights = weights.mean(dim=1)
        return output, weights

    return wrapped
