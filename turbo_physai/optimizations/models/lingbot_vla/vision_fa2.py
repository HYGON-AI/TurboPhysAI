# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections import OrderedDict

import torch
from flash_attn import flash_attn_func

from lingbotvla.models.vla.pi0 import qwenvl_in_vla
from lingbotvla.models.vla.pi0.qwenvl_in_vla import flash_attn_varlen_func

logger = logging.getLogger(__name__)


_UNIFORM_CACHE: "OrderedDict[tuple, bool]" = OrderedDict()
_UNIFORM_CACHE_SIZE = 128


def _is_uniform(cu_seqlens, nseg: int, seglen: int) -> bool:
    key = (nseg, seglen, tuple(cu_seqlens.detach().cpu().tolist()))
    hit = _UNIFORM_CACHE.get(key)
    if hit is not None:
        _UNIFORM_CACHE.move_to_end(key)
        return hit
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    ok = bool(torch.equal(lens, torch.full_like(lens, seglen)))
    _UNIFORM_CACHE[key] = ok
    if len(_UNIFORM_CACHE) > _UNIFORM_CACHE_SIZE:
        _UNIFORM_CACHE.popitem(last=False)
    logger.debug(
        "lingbot-vla: vision FA2 fixed-length path %s (nseg=%d, seglen=%d)", ok, nseg, seglen
    )
    return ok


@torch._dynamo.disable
def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
    seq_length = hidden_states.shape[0]
    q, k, v = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )

    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    else:
        cos, sin = position_embeddings

    q, k = qwenvl_in_vla.apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    q = q.squeeze(0)
    k = k.squeeze(0)

    out_fp32_atten = k.dtype == torch.float32
    if out_fp32_atten:
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

    nseg = cu_seqlens.numel() - 1
    seglen = seq_length // nseg
    if seglen * nseg == seq_length and _is_uniform(cu_seqlens, nseg, seglen):
        attn_output = flash_attn_func(
            q.reshape(nseg, seglen, self.num_heads, -1),
            k.reshape(nseg, seglen, self.num_heads, -1),
            v.reshape(nseg, seglen, self.num_heads, -1),
        ).reshape(seq_length, -1)
    else:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(seq_length, -1)

    if out_fp32_atten:
        attn_output = attn_output.to(torch.float32)
    return self.proj(attn_output)
