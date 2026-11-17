# Copyright 2026 Hygon Information Technology Co., Ltd.
# Portions adapted from PyTorch; the original copyright notices and license are
# preserved in third_party/licenses/PyTorch-BSD-3-Clause.txt.
# SPDX-License-Identifier: BSD-3-Clause
"""Reference ACT softmax/dropout attention candidate, installed by Turbo.

Adapted from the reference LeRobot-ACT ``act_attention_fusion.py`` and the
math branch of PyTorch ``torch.nn.functional.multi_head_attention_forward``
(PyTorch, BSD-3-Clause).  The only target installed by this module is the
declared functional MHA entry; unsupported calls return to the original.
"""

import functools
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_REGISTER_LENGTH = 902


def _native_probs(scores, p):
    return F.dropout(torch.softmax(scores, dim=-1), p=p, training=True)


_compiled_probs = None


def _compiled_softmax_dropout(scores, p):
    if torch.compiler.is_compiling():
        return _native_probs(scores, p)
    global _compiled_probs
    if _compiled_probs is None:
        _compiled_probs = torch.compile(
            _native_probs, dynamic=False, mode="max-autotune-no-cudagraphs", fullgraph=True
        )
    return _compiled_probs(scores, p)


@triton.jit
def _softmax_dropout_backward_kernel(P, G, M, O, N: tl.constexpr, BLOCK: tl.constexpr, SCALE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offsets = row * N + cols
    valid = cols < N
    p = tl.load(P + offsets, valid, 0)
    g = tl.load(G + offsets, valid, 0)
    mask = tl.load(M + offsets, valid, 0)
    d = tl.where(mask, g * SCALE, 0.0)
    tl.store(O + offsets, p * (d - tl.sum(p * d, 0)), valid)


class _RegisterSoftmaxDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, probability):
        probs = torch.softmax(scores, dim=-1).contiguous()
        output, mask = torch.native_dropout(probs, probability, True)
        ctx.save_for_backward(probs, mask)
        ctx.scale = 1.0 / (1.0 - probability)
        return output

    @staticmethod
    def backward(ctx, gradient):
        probs, mask = ctx.saved_tensors
        if torch.is_grad_enabled():
            raise RuntimeError("ACT register softmax-dropout supports first-order gradients only")
        length = probs.shape[-1]
        output = torch.empty_like(probs)
        _softmax_dropout_backward_kernel[(probs.numel() // length,)](
            probs, gradient.contiguous(), mask, output, length,
            triton.next_power_of_2(length), ctx.scale, num_warps=8,
        )
        return output, None


def _attention_probs(scores, probability, register_backward):
    if (
        register_backward and scores.is_cuda and scores.dtype == torch.float32
        and scores.shape[-1] == _REGISTER_LENGTH and 0.0 < probability < 1.0
    ):
        return _RegisterSoftmaxDropout.apply(scores, probability)
    return _compiled_softmax_dropout(scores, probability)


def _project_qkv(query, key, value, weight, bias):
    """Project Q/K/V through the stock packed-projection implementation."""
    return F._in_projection_packed(query, key, value, weight, bias)


def _unmasked_qk(query, key):
    """Compute unmasked QK scores; replaceable by the QK TF32 Group."""
    return torch.bmm(query, key)


def _math_attention(
    query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
    bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
    training=True, key_padding_mask=None, need_weights=True, attn_mask=None,
    use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
    v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True,
    is_causal=False, *, probability_fn,
):
    # Mirrors the supported batched, packed-projection, need_weights math path.
    del bias_k, bias_v, add_zero_attn, training, need_weights, attn_mask
    del use_separate_proj_weight, q_proj_weight, k_proj_weight, v_proj_weight
    del static_k, static_v, is_causal
    target_len, batch, embed_dim = query.shape
    source_len = key.shape[0]
    head_dim = embed_dim_to_check // num_heads
    q, k, v = _project_qkv(query, key, value, in_proj_weight, in_proj_bias)
    q = q.view(target_len, batch * num_heads, head_dim).transpose(0, 1)
    k = k.view(source_len, batch * num_heads, head_dim).transpose(0, 1)
    v = v.view(source_len, batch * num_heads, head_dim).transpose(0, 1)
    q_scaled = q * math.sqrt(1.0 / float(head_dim))
    if key_padding_mask is None:
        scores = _unmasked_qk(q_scaled, k.transpose(-2, -1))
    else:
        mask = F._canonical_mask(
            mask=key_padding_mask, mask_name="key_padding_mask", other_type=None,
            other_name="", target_type=query.dtype, check_other=False,
        )
        mask = mask.view(batch, 1, 1, source_len).expand(-1, num_heads, -1, -1)
        mask = mask.reshape(batch * num_heads, 1, source_len)
        scores = torch.baddbmm(mask, q_scaled, k.transpose(-2, -1))
    probs = probability_fn(scores, dropout_p)
    output = torch.bmm(probs, v).transpose(0, 1).reshape(target_len * batch, embed_dim)
    output = F.linear(output, out_proj_weight, out_proj_bias)
    output = output.view(target_len, batch, output.size(1))
    weights = probs.view(batch, num_heads, target_len, source_len)
    if average_attn_weights:
        weights = weights.mean(dim=1)
    return output, weights


def attention_softmax_dropout_wrapper(original, options):
    """Engine wrap factory for ``torch.nn.functional.multi_head_attention_forward``."""
    register_backward = options.get("register_backward", True)
    if not isinstance(register_backward, bool):
        raise TypeError("register_backward must be bool")

    @functools.wraps(original)
    def wrapped(
        query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
        bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias,
        training=True, key_padding_mask=None, need_weights=True, attn_mask=None,
        use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None,
        v_proj_weight=None, static_k=None, static_v=None, average_attn_weights=True,
        is_causal=False,
    ):
        args = (query, key, value, embed_dim_to_check, num_heads, in_proj_weight,
                in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p,
                out_proj_weight, out_proj_bias)
        kwargs = dict(training=training, key_padding_mask=key_padding_mask,
                      need_weights=need_weights, attn_mask=attn_mask,
                      use_separate_proj_weight=use_separate_proj_weight,
                      q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                      v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v,
                      average_attn_weights=average_attn_weights, is_causal=is_causal)
        supported = (
            query.is_cuda and query.dtype == torch.float32 and training and need_weights
            and not is_causal and attn_mask is None and not add_zero_attn
            and bias_k is None and bias_v is None and static_k is None and static_v is None
            and not use_separate_proj_weight and in_proj_weight is not None
            and key.shape == value.shape and query.shape[1] == key.shape[1]
            and query.shape[2] == key.shape[2] == embed_dim_to_check
            and 0.0 <= dropout_p <= 1.0
            and (key_padding_mask is None or key_padding_mask.shape == (key.shape[1], key.shape[0]))
        )
        if not supported:
            return original(*args, **kwargs)
        return _math_attention(
            *args, **kwargs,
            probability_fn=lambda scores, p: _attention_probs(scores, p, register_backward),
        )

    return wrapped
