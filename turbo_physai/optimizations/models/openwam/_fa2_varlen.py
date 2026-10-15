# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""Skip DAS offset filtering only for offsets constructed by our padding packer."""

import hashlib
import inspect
import logging

import torch

logger = logging.getLogger(__name__)

# Private FA2 entry points are not a stable API. Gate the complete autograd
# implementation we inspected; other builds retain the public FA2 entry point.
_SUPPORTED = (
    "8c3284214b54243399620f8cc2fb01cdb9c9a3ce4d1c3997661429f62cb7bc74",
    "a9d6d29d7723e7be551f1906f9f7295188d65b4ab26c36c3db7870493d16036a",
)


def packed_varlen_kernel(public_kernel):
    if not getattr(torch.version, "hip", None):
        return public_kernel
    try:
        from flash_attn import flash_attn_interface as backend

        original = backend.FlashAttnVarlenFunc
        fingerprints = tuple(
            hashlib.sha256(
                inspect.getsource(fn).replace("\r\n", "\n").encode()
            ).hexdigest()
            for fn in (original.forward, original.backward)
        )
        if fingerprints != _SUPPORTED:
            logger.info(
                "[optimizations] FA2 padding retains public entry: unrecognized autograd implementation"
            )
            return public_kernel
        forward = backend._wrapped_flash_attn_varlen_forward
        backward = original.backward
    except (ImportError, AttributeError, OSError, TypeError):
        logger.info(
            "[optimizations] FA2 padding retains public entry: private API unavailable"
        )
        return public_kernel

    class PackedVarlen(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k):
            scale = q.shape[-1] ** -0.5
            out, lse, _, rng = forward(
                q,
                k,
                v,
                None,
                cu_q,
                cu_k,
                None,
                None,
                None,
                None,
                max_q,
                max_k,
                0.0,
                scale,
                False,
                False,
                -1,
                -1,
                softcap=0.0,
                return_softmax=False,
            )
            if any(ctx.needs_input_grad[:3]):
                ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k, rng)
                ctx.qk_headdim = q.shape[-1]
                ctx.dropout_p = 0.0
                ctx.max_seqlen_q, ctx.max_seqlen_k = max_q, max_k
                ctx.softmax_scale = scale
                ctx.causal = False
                ctx.window_size = (-1, -1)
                ctx.softcap = 0.0
                ctx.alibi_slopes = None
                ctx.deterministic = False
            return out

        @staticmethod
        def backward(ctx, dout):
            # Retain the installed vendor backward, including its head slicing.
            grads = backward(ctx, dout)
            return (*grads[:3], None, None, None, None)

    def run(q, k, v, cu_q, cu_k, max_q, max_k, *, dropout_p=0.0, causal=False):
        if dropout_p != 0.0 or causal:
            return public_kernel(
                q, k, v, cu_q, cu_k, max_q, max_k, dropout_p=dropout_p, causal=causal
            )
        # Private to fa2_padding: cu_q ends at len(q), cu_k at len(k), including
        # dummy keys for empty rows. No data-dependent GPU validation is needed.
        return PackedVarlen.apply(q, k, v, cu_q, cu_k, max_q, max_k)

    logger.info(
        "[optimizations] FA2 padding uses validated packed offsets without duplicate filtering"
    )
    return run
