"""Wan text encoder (T5): SDPA attention kernel + torch.compile'd forward.

"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F


def t5_attention_forward(self, x, context=None, mask=None, pos_bias=None):
    """
    x:          [B, L1, C].
    context:    [B, L2, C] or None.
    mask:       [B, L2] or [B, L1, L2] or None.
    """
    # check inputs
    context = x if context is None else context
    b, n, c = x.size(0), self.num_heads, self.head_dim

    # compute query, key, value, as [B, num_heads, L, head_dim] for sdpa
    q = self.q(x).view(b, -1, n, c).transpose(1, 2)
    k = self.k(context).view(b, -1, n, c).transpose(1, 2)
    v = self.v(context).view(b, -1, n, c).transpose(1, 2)

    # attention bias
    attn_bias = None
    if pos_bias is not None or mask is not None:
        attn_bias = x.new_zeros(b, n, q.size(2), k.size(2))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1,
                              -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

    # compute attention (T5 does not use scaling)
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=1.0)

    # output
    x = x.transpose(1, 2).reshape(b, -1, n * c)
    x = self.o(x)
    x = self.dropout(x)
    return x


_COMPILE_DEFAULTS = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}


def compile_forward(original, options):
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    compile_kwargs = {**_COMPILE_DEFAULTS, **(options or {})}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped
