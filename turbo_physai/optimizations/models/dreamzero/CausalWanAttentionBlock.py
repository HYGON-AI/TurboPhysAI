"""Replace CausalWanAttentionBlock.forward with a version whose modulation/FFN
tensor regions run through torch.compile'd functions, and wrap __init__ to
torch.compile the whole WanI2VCrossAttention cross-attention module
(baseline model commit ab790c198fbce33503358efbbd4187ce9a89adf3).

_prepare_self_attention and _ffn_residual are compiled once at import time --
no extra config flag or "already compiled" instance guard needed.
"""

from __future__ import annotations

import functools

import torch

_COMPILE_DEFAULTS = {"fullgraph": True, "dynamic": True}


def _prepare_self_attention(self, x, shift, scale):
    return self.norm1(x) * (1 + scale) + shift


_prepare_self_attention = torch.compile(_prepare_self_attention, **_COMPILE_DEFAULTS)


def _ffn_residual(self, x, shift, scale, gate):
    y = self.ffn(self.norm2(x) * (1 + scale) + shift)
    return x + y * gate


_ffn_residual = torch.compile(_ffn_residual, **_COMPILE_DEFAULTS)


def compile_cross_attn(original, options):
    from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
        WanI2VCrossAttention,
    )

    compile_kwargs = dict(_COMPILE_DEFAULTS)
    compile_kwargs.update(options or {})

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if isinstance(self.cross_attn, WanI2VCrossAttention):
            self.cross_attn.compile(**compile_kwargs)

    return __init__


def forward(
    self,
    x: torch.Tensor,
    e: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    context: torch.Tensor,
    kv_cache: torch.Tensor | None = None,
    crossattn_cache: torch.Tensor | None = None,
    current_start_frame: int = 0,
    is_tf: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, F, 6, C]
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    e = self.modulation.unsqueeze(1) + e

    # Align outside compile, before splitting, to keep modulation channels
    # packed consistently at the region boundaries.
    L = x.shape[1]
    L_e = e.shape[1]
    if L_e > L:
        e = e[:, :L]
    elif L_e < L:
        repeat = (L + L_e - 1) // L_e
        e = e.repeat_interleave(repeat, dim=1)[:, :L]
    e = e.unbind(dim=2)

    # self-attention
    y, updated_kv_cache = self.self_attn(
        x=_prepare_self_attention(self, x, e[0], e[1]),
        freqs=freqs,
        freqs_action=freqs_action,
        freqs_state=freqs_state,
        action_register_length=action_register_length,
        kv_cache=kv_cache,
        is_tf=is_tf,
        current_start_frame=current_start_frame,
    )
    x = x + (y * e[2])

    # I2V cross-attention has its own whole-module compile entry point.
    x = x + self.cross_attn(self.norm3(x), context)
    x = _ffn_residual(self, x, e[3], e[4], e[5])
    return x, updated_kv_cache
