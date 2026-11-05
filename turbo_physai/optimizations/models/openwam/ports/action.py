# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import torch
import torch.nn.functional as F
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
    run_attention,
    run_ffn,
    run_norm,
    run_projection,
)


def _modulate(x, shift, scale):
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


def act_post(block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, attn_out, context, text_mask, attention):
    sa, ca = block.self_attn, block.cross_attn
    n = block.context_attn_norm
    from openwam.model.kernels.fused_blocks import _compiled, _wan_post_a
    hs, cq = _compiled("act_post_a", _wan_post_a)(
        residual_x,
        gate_msa,
        attn_out,
        sa.o.weight,
        sa.o.bias,
        n.weight,
        n.bias,
        ca.q.weight,
        ca.q.bias,
        ca.norm_q.weight,
        float(n.eps),
        float(ca.norm_q.eps),
    )
    ck = ca.norm_k(ca.k(context))
    cv = ca.v(context)
    cx = run_attention(
        attention,
        cq,
        ck,
        cv,
        ca.num_heads,
        text_mask,
        label="action_cross_attention",
    )
    if active_ffn_boundary():
        hs = hs + F.linear(cx, ca.o.weight, ca.o.bias)
        m = F.layer_norm(hs, (hs.shape[-1],), None, None, float(block.ffn_norm.eps)) * (1 + scale_mlp) + shift_mlp
        ffn_out = run_ffn(block.ffn, m, video=False)
        return hs + gate_mlp * ffn_out
    from openwam.model.kernels.fused_blocks import _post_b
    f1, f2 = block.ffn[0], block.ffn[2]
    return _compiled("act_post_b", _post_b)(
        hs,
        cx,
        ca.o.weight,
        ca.o.bias,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        f1.weight,
        f1.bias,
        f2.weight,
        f2.bias,
        float(block.ffn_norm.eps),
        None,
        None,
    )


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

act_post_wrapper = _port_wrapper(
    act_post,
    "openwam.model.kernels.fused_blocks.act_post",
    "openwam.compute",
)
