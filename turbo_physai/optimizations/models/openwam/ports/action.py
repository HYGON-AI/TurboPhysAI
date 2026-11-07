# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import torch
from openwam.model.kernels import fused_blocks
from openwam.model.action_backbone.separate_action_dit import (
    get_attention_fn,
    rearrange,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.action_backbone.separate_action_dit import (
        ActionDiTState,
        ActionState,
        Optional,
    )
from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.attention import fa2_padding
from turbo_physai.optimizations.models.openwam.sac import (
    active_ffn_boundary,
    ffn_boundary,
    run_attention,
    run_ffn,
    run_norm,
    run_projection,
)


def _modulate(x, shift, scale):
    from turbo_physai.optimizations.models.openwam import pointwise
    from turbo_physai.optimizations.models.openwam.sac import extras_enabled

    if enabled("pointwise_compile") or extras_enabled():
        return pointwise.modulate(x, shift, scale)
    return x * (1 + scale) + shift


def BridgeCrossAttention_forward(
    self,
    x_action: torch.Tensor,
    x_video: torch.Tensor,
    ctx_mask: Optional[torch.Tensor] = None,
):
    q = run_norm(self.norm_q, run_projection(self.q, x_action))
    k = run_norm(self.norm_k, run_projection(self.k, x_video))
    v = run_projection(self.v, x_video)
    x = run_attention(
        functools.partial(BridgeCrossAttention__attend, self),
        q,
        k,
        v,
        ctx_mask,
        label="action_cross_attention",
    )
    return run_projection(self.o, x)


def BridgeCrossAttention__attend(self, q, k, v, ctx_mask):
    padded = fa2_padding(q, k, v, ctx_mask, self.num_heads)
    if padded is not None:
        return padded
    q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
    if ctx_mask is None:
        x = get_attention_fn()(q, k, v)
    else:
        if ctx_mask.dim() == 3:
            ctx_mask = ctx_mask.unsqueeze(1)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=ctx_mask
        )
    x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
    return x


def CrossAttnActionDiTBlock_forward(
    self,
    x_action,
    x_video,
    t_mod,
    freqs: Optional[torch.Tensor] = None,
    context: Optional[torch.Tensor] = None,
    context_mask: Optional[torch.Tensor] = None,
):
    (
        shift_sa,
        scale_sa,
        gate_sa,
        shift_ca,
        scale_ca,
        gate_ca,
        shift_ff,
        scale_ff,
        gate_ff,
    ) = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
        9, dim=1
    )

    h = _modulate(self.self_attn_norm(x_action), shift_sa, scale_sa)
    x_action = x_action + gate_sa * self.self_attn(h, freqs=freqs)

    h = _modulate(self.bridge_attn_norm(x_action), shift_ca, scale_ca)
    x_action = x_action + gate_ca * self.cross_attn(h, x_video)

    if context is not None:
        x_action = x_action + self.context_attn(
            self.context_attn_norm(x_action), context, ctx_mask=context_mask
        )

    h = _modulate(self.ffn_norm(x_action), shift_ff, scale_ff)
    x_action = x_action + gate_ff * self.ffn(h)

    return x_action


def SelfAttnActionDiTBlock_forward(
    self,
    x: torch.Tensor,
    context: Optional[torch.Tensor],
    t_mod: torch.Tensor,
    freqs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference forward used in equivalence tests; not invoked by the MoT driver."""
    chunks = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(
        6, dim=1
    )
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    attn_input = _modulate(self.self_attn_norm(x), shift_msa, scale_msa)
    x = self.gate(x, gate_msa, self.self_attn(attn_input, freqs=freqs))
    if context is not None:
        x = x + self.cross_attn(self.context_attn_norm(x), context)
    mlp_input = _modulate(self.ffn_norm(x), shift_mlp, scale_mlp)
    x = self.gate(x, gate_mlp, self.ffn(mlp_input))
    return x


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

BridgeCrossAttention_forward_wrapper = _port_wrapper(
    BridgeCrossAttention_forward,
    "openwam.model.action_backbone.separate_action_dit.BridgeCrossAttention.forward",
    "openwam.compute",
)

CrossAttnActionDiTBlock_forward_wrapper = _port_wrapper(
    CrossAttnActionDiTBlock_forward,
    "openwam.model.action_backbone.separate_action_dit.CrossAttnActionDiTBlock.forward",
    "openwam.compute",
)

SelfAttnActionDiTBlock_forward_wrapper = _port_wrapper(
    SelfAttnActionDiTBlock_forward,
    "openwam.model.action_backbone.separate_action_dit.SelfAttnActionDiTBlock.forward",
    "openwam.compute",
)


def rope_apply_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D rotary position embedding to Q or K.

    Mirrors FastWAM/Wan RoPE numerics by doing the complex multiply through a
    float64 complex view, then restoring the caller's dtype.

    Args:
        x:     (B, H, S, D) head-split tensor.
        freqs: (S, D // 2) complex frequencies from ``precompute_freqs_cis_1d``.

    Returns:
        Rotated tensor with the same dtype and shape as ``x``.
    """
    from turbo_physai.optimizations.models.openwam import pointwise

    if enabled("rope_real"):
        return pointwise.rotate(x, freqs.view(1, 1, x.shape[-2], -1))
    x_c = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    freqs = freqs.to(x_c.device).view(1, 1, x_c.shape[-2], x_c.shape[-1])
    return torch.view_as_real(x_c * freqs).flatten(-2).to(x.dtype)


def SelfAttnActionDiTBlock_gate(
    self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor
) -> torch.Tensor:
    from turbo_physai.optimizations.models.openwam import pointwise
    from turbo_physai.optimizations.models.openwam.sac import extras_enabled

    if enabled("pointwise_compile") or extras_enabled():
        return pointwise.gate(x, gate, residual)
    return x + gate * residual


SelfAttnActionDiTBlock_gate_wrapper = _port_wrapper(
    SelfAttnActionDiTBlock_gate,
    "openwam.model.action_backbone.separate_action_dit.SelfAttnActionDiTBlock.gate",
    "openwam.compute",
)


def ActionSelfAttention_forward(self, x, freqs: Optional[torch.Tensor] = None):
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    if freqs is not None:
        from .dit import fused_rope_ok, rope_apply_auto
        from .dit import rope_apply

        if fused_rope_ok(q, freqs, self.num_heads):
            q = rope_apply_auto(q, freqs, self.num_heads, rope_apply)
            k = rope_apply_auto(k, freqs, self.num_heads, rope_apply)
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
            q = rope_apply_1d(q, freqs)
            k = rope_apply_1d(k, freqs)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)
    x = get_attention_fn()(q, k, v)
    x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
    return self.o(x)


ActionSelfAttention_forward_wrapper = _port_wrapper(
    ActionSelfAttention_forward,
    "openwam.model.action_backbone.separate_action_dit.ActionSelfAttention.forward",
    "openwam.compute",
)


def ActionDiT_pre_attn_at_layer_for_compile(
    self, layer_id: int, astate: "ActionState", *, block=None
):
    """Compile-friendly pre-attention half using a tensor tuple post-state.

    Returns a 4-tuple ``(q, k, v, post_state)`` where ``post_state`` is the
    tensor tuple ``(residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)``.
    """
    payload: ActionDiTState = astate.payload
    if block is None:
        block = self.blocks[layer_id]
    residual_x = payload.x_action

    if fused_blocks.act_block_supported(block, payload.t_mod):
        q, k, v, gate_msa, shift_mlp, scale_mlp, gate_mlp = fused_blocks.act_pre(
            block, residual_x, payload.t_mod
        )
    else:
        chunks = (
            block.modulation.to(dtype=payload.t_mod.dtype, device=payload.t_mod.device)
            + payload.t_mod
        ).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

        attn_input = _modulate(
            run_norm(block.self_attn_norm, residual_x), shift_msa, scale_msa
        )

        sa = block.self_attn
        q = run_norm(sa.norm_q, run_projection(sa.q, attn_input))
        k = run_norm(sa.norm_k, run_projection(sa.k, attn_input))
        v = run_projection(sa.v, attn_input)

    freqs = payload.action_freqs
    if freqs is not None:
        from .dit import fused_rope_ok, rope_apply_auto
        from .dit import rope_apply

        if fused_rope_ok(q, freqs, self._num_heads):
            q_out = rope_apply_auto(q, freqs, self._num_heads, rope_apply)
            k_out = rope_apply_auto(k, freqs, self._num_heads, rope_apply)
            v_out = v
        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=self._num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=self._num_heads)
            q = rope_apply_1d(q, freqs)
            k = rope_apply_1d(k, freqs)
            q_out = rearrange(q, "b n s d -> b s (n d)", n=self._num_heads)
            k_out = rearrange(k, "b n s d -> b s (n d)", n=self._num_heads)
            v_out = v
    else:
        q_out, k_out, v_out = q, k, v

    post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    return q_out, k_out, v_out, post_state


ActionDiT_pre_attn_at_layer_for_compile_wrapper = _port_wrapper(
    ActionDiT_pre_attn_at_layer_for_compile,
    "openwam.model.action_backbone.separate_action_dit.ActionDiT.pre_attn_at_layer_for_compile",
    "openwam.compute",
)

rope_apply_1d_wrapper = _port_wrapper(
    rope_apply_1d,
    "openwam.model.action_backbone.components.rope_apply_1d",
    "openwam.compute",
)


def ActionDiT_post_attn_at_layer_for_compile(
    self,
    layer_id: int,
    astate: "ActionState",
    attn_out: torch.Tensor,
    post_state: tuple[torch.Tensor, ...],
    *,
    block=None,
) -> "ActionState":
    """Compile-friendly post-attention half consuming a tensor tuple."""
    payload: ActionDiTState = astate.payload
    if block is None:
        block = self.blocks[layer_id]
    residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

    if payload.context is not None and fused_blocks.act_block_supported(
        block, payload.t_mod
    ):
        text_mask = payload.context_mask
        if text_mask is not None:
            if text_mask.dim() == 2:
                text_mask = text_mask.unsqueeze(1).expand(-1, residual_x.shape[1], -1)
            elif text_mask.dim() not in (3, 4):
                raise ValueError(
                    "ActionDiTState.context_mask must be [B, L], [B, T_action, L], "
                    f"or broadcastable [B, heads, T_action, L], got {tuple(text_mask.shape)}"
                )
        payload.x_action = fused_blocks.act_post(
            block,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            attn_out,
            payload.context,
            text_mask,
            fused_blocks.masked_mha,
        )
        return astate

    x = block.gate(residual_x, gate_msa, run_projection(block.self_attn.o, attn_out))
    if payload.context is not None:
        text_mask = payload.context_mask
        if text_mask is not None:
            if text_mask.dim() == 2:
                text_mask = text_mask.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif text_mask.dim() not in (3, 4):
                raise ValueError(
                    "ActionDiTState.context_mask must be [B, L], [B, T_action, L], "
                    f"or broadcastable [B, heads, T_action, L], got {tuple(text_mask.shape)}"
                )
        x = x + block.cross_attn(
            run_norm(block.context_attn_norm, x), payload.context, ctx_mask=text_mask
        )
    mlp_input = _modulate(run_norm(block.ffn_norm, x), shift_mlp, scale_mlp)
    use_ffn_boundary = active_ffn_boundary() or (
        ffn_boundary() and torch.compiler.is_compiling()
    )
    mlp_output = (
        run_ffn(block.ffn, mlp_input) if use_ffn_boundary else block.ffn(mlp_input)
    )
    x = block.gate(x, gate_mlp, mlp_output)

    payload.x_action = x
    return astate


ActionDiT_post_attn_at_layer_for_compile_wrapper = _port_wrapper(
    ActionDiT_post_attn_at_layer_for_compile,
    "openwam.model.action_backbone.separate_action_dit.ActionDiT.post_attn_at_layer_for_compile",
    "openwam.compute",
)
