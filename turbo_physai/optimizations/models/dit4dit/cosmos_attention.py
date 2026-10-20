# Copyright 2025 The NVIDIA Team and The HuggingFace Team.
# Copyright 2026 DiT4DiT team.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified for TurboPhysAI; see THIRD_PARTY_NOTICES.md.
# Licensed under the Apache License, Version 2.0.

"""Targeted launch-overhead optimizations for the Diffusers Cosmos transformer."""

from __future__ import annotations


import torch
from torch import nn

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb


class CosmosAttnProcessor2_0SkipNoopRepeat:
    """CosmosAttnProcessor2_0 without materializing repeat_interleave(1)."""

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(
                query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2
            )
            key = apply_rotary_emb(
                key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2
            )

        query_width = query.size(3)
        key_width = key.size(3)
        value_width = value.size(3)
        key_repeat = query_width // key_width
        value_repeat = query_width // value_width
        # The stock processor always calls repeat_interleave.  Cosmos 2.5 uses
        # equal Q/K/V head widths here, so that call only creates a redundant
        # device copy.  Preserve the original behavior for actual GQA ratios.
        if key_repeat != 1:
            key = key.repeat_interleave(key_repeat, dim=3)
        if value_repeat != 1:
            value = value.repeat_interleave(value_repeat, dim=3)

        hidden_states = dispatch_attention_fn(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CachedCosmosRotaryPosEmbed(nn.Module):
    """Cache parameter-free Cosmos RoPE tensors for repeated input shapes.

    CosmosRotaryPosEmbed only reads the input shape and device. Its result is
    always FP32 and has no autograd dependency on ``hidden_states``, so the
    detached tensors can safely be reused across training micro-steps.
    """

    def __init__(self, rope: nn.Module, max_cache_entries: int = 4) -> None:
        super().__init__()
        self.rope = rope
        self.max_cache_entries = max_cache_entries
        # Keep cached tensors out of state_dict and DDP/DeepSpeed broadcasts.
        self._rope_cache: dict[
            tuple[int, int, int, int | None, str],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def forward(
        self, hidden_states: torch.Tensor, fps: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        key = (num_frames, height, width, fps, str(hidden_states.device))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached

        # RoPE depends on constants, shape, fps and device only. Avoid building
        # an autograd graph during the one-time cache miss.
        with torch.no_grad():
            cos, sin = self.rope(hidden_states, fps=fps)
        cached = (cos.detach(), sin.detach())

        if len(self._rope_cache) >= self.max_cache_entries:
            # Training uses one fixed shape. Bound stale GPU memory if another
            # shape is introduced by validation or a later configuration.
            self._rope_cache.clear()
        self._rope_cache[key] = cached
        return cached


def install_cosmos_rope_cache(
    transformer: torch.nn.Module, max_cache_entries: int = 4
) -> bool:
    """Wrap ``transformer.rope`` with a bounded cache; return True if active."""
    rope = getattr(transformer, "rope", None)
    if rope is None:
        return False
    if isinstance(rope, CachedCosmosRotaryPosEmbed):
        return True
    if rope.__class__.__name__ != "CosmosRotaryPosEmbed":
        return False
    transformer.rope = CachedCosmosRotaryPosEmbed(
        rope, max_cache_entries=max_cache_entries
    )
    return True
