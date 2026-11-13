# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""RoPE：一次 forward 只算一遍 cos/sin，旋转交给 compile 融合。"""

from __future__ import annotations

import weakref

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

_CACHE = {}
_COMPILED = None


def _compiled_apply(query_states, key_states, cos, sin):
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(apply_rotary_pos_emb, dynamic=False)
    return _COMPILED(query_states, key_states, cos, sin, unsqueeze_dim=2)


def _position_embeddings(self, query_states, position_ids):
    """position_ids 复用 cos/sin"""
    entry = _CACHE.get("entry")
    if (
        entry is not None
        and entry["ref"]() is position_ids
        and entry["dtype"] is query_states.dtype
        and entry["device"] == query_states.device
    ):
        return entry["cos"], entry["sin"]

    cos, sin = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
    _CACHE["entry"] = {
        "ref": weakref.ref(position_ids),
        "dtype": query_states.dtype,
        "device": query_states.device,
        "cos": cos,
        "sin": sin,
    }
    return cos, sin


def apply_mrope(self, query_states, key_states, position_ids):
    cos, sin = _position_embeddings(self, query_states, position_ids)
    if torch.compiler.is_compiling():
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
    return _compiled_apply(query_states, key_states, cos, sin)
