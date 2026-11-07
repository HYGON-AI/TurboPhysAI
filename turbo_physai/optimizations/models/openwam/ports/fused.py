# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM 0ca62ddbc3052252351aa28df88af1607f77381d.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from openwam.model.kernels.fused_blocks import (
    _compiled,
    _rms,
    _post_b,
    _attention_forward_supported,
    _ffn_forward_supported,
)
from ..settings import port_wrapper as _port_wrapper


def _wan_post_a(
    residual_x,
    gate_msa,
    attn_out,
    o_w,
    o_b,
    n3_w,
    n3_b,
    cq_w,
    cq_b,
    cnq_w,
    eps3: float,
    eps_q: float,
    context=None,
    ck_w=None,
    ck_b=None,
    cnk_w=None,
    cv_w=None,
    cv_b=None,
    eps_k: float = 1e-6,
):
    hs = residual_x + gate_msa * F.linear(attn_out, o_w, o_b)
    h3 = F.layer_norm(hs, (hs.shape[-1],), n3_w, n3_b, eps3)
    cq = _rms(F.linear(h3, cq_w, cq_b), cnq_w, eps_q)
    if context is None:
        return hs, cq
    ck = _rms(F.linear(context, ck_w, ck_b), cnk_w, eps_k)
    cv = F.linear(context, cv_w, cv_b)
    return hs, cq, ck, cv


def _post_b_pre_ffn(hs, cx, co_w, co_b, shift_mlp, scale_mlp, eps2: float):
    hs = hs + F.linear(cx, co_w, co_b)
    m = (
        F.layer_norm(hs, (hs.shape[-1],), None, None, eps2) * (1 + scale_mlp)
        + shift_mlp
    )
    return hs, m


def _post_b_gate(hs, gate_mlp, ffn_out):
    return hs + gate_mlp * ffn_out


def wan_post(
    block,
    residual_x,
    gate_msa,
    shift_mlp,
    scale_mlp,
    gate_mlp,
    attn_out,
    context,
    context_mask,
    attention,
):
    sa, ca = block.self_attn, block.cross_attn
    hs, cq, ck, cv = _compiled("wan_post_a", _wan_post_a)(
        residual_x,
        gate_msa,
        attn_out,
        sa.o.weight,
        sa.o.bias,
        block.norm3.weight,
        block.norm3.bias,
        ca.q.weight,
        ca.q.bias,
        ca.norm_q.weight,
        float(block.norm3.eps),
        float(ca.norm_q.eps),
        context,
        ca.k.weight,
        ca.k.bias,
        ca.norm_k.weight,
        ca.v.weight,
        ca.v.bias,
        float(ca.norm_k.eps),
    )
    from turbo_physai.optimizations.models.openwam.sac import (
        active_ffn_boundary,
        run_attention,
        run_ffn,
    )

    cx = run_attention(
        attention, cq, ck, cv, ca.num_heads, context_mask, label="video_cross_attention"
    )

    if active_ffn_boundary():
        hs, m = _compiled("wan_post_b_pre", _post_b_pre_ffn)(
            hs, cx, ca.o.weight, ca.o.bias, shift_mlp, scale_mlp, float(block.norm2.eps)
        )
        ffn_out = run_ffn(block.ffn, m, video=True)
        return _compiled("wan_post_b_gate", _post_b_gate)(hs, gate_mlp, ffn_out)

    f1, f2 = block.ffn[0], block.ffn[2]
    return _compiled("post_b", _post_b)(
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
        float(block.norm2.eps),
        None,
        None,
    )


wan_post_wrapper = _port_wrapper(
    wan_post, "openwam.model.kernels.fused_blocks.wan_post", "openwam.compute"
)


def act_post(
    block,
    residual_x,
    gate_msa,
    shift_mlp,
    scale_mlp,
    gate_mlp,
    attn_out,
    context,
    text_mask,
    attention,
):
    sa, ca = block.self_attn, block.cross_attn
    n = block.context_attn_norm
    hs, cq, ck, cv = _compiled("act_post_a", _wan_post_a)(
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
        context,
        ca.k.weight,
        ca.k.bias,
        ca.norm_k.weight,
        ca.v.weight,
        ca.v.bias,
        float(ca.norm_k.eps),
    )
    from turbo_physai.optimizations.models.openwam.sac import (
        active_ffn_boundary,
        run_attention,
        run_ffn,
    )

    cx = run_attention(
        attention, cq, ck, cv, ca.num_heads, text_mask, label="action_cross_attention"
    )

    if active_ffn_boundary():
        hs, m = _compiled("act_post_b_pre", _post_b_pre_ffn)(
            hs,
            cx,
            ca.o.weight,
            ca.o.bias,
            shift_mlp,
            scale_mlp,
            float(block.ffn_norm.eps),
        )
        ffn_out = run_ffn(block.ffn, m, video=False)
        return _compiled("act_post_b_gate", _post_b_gate)(hs, gate_mlp, ffn_out)

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


act_post_wrapper = _port_wrapper(
    act_post, "openwam.model.kernels.fused_blocks.act_post", "openwam.compute"
)


def _standard_forward(module, expected_type, allow_sac: bool = False) -> bool:
    """Only inline the shipped forward, preserving wrappers and module hooks.

    ``requires_grad_(False)`` still permits gradients to inputs, and ``eval()``
    does not disable autograd. Neither is a reason to reject fusion. In contrast,
    ``freeze_modules`` wraps every descendant's forward in no_grad; reading its
    weights directly would bypass that boundary. Check live metadata so freezing
    or replacing a submodule after a previous fast call cannot leave a stale yes.
    """
    attrs = module.__dict__
    has_forward = "forward" in attrs
    if (
        has_forward
        and allow_sac
        and (
            attrs.get("_openwam_sac_wrapped", False)
            or getattr(getattr(module, "forward", None), "_openwam_sac_wrapped", False)
        )
    ):
        has_forward = False
    type_match = (
        type(module) in expected_type
        if isinstance(expected_type, tuple)
        else type(module) is expected_type
    )
    return (
        type_match
        and not attrs.get("_openwam_no_grad_wrapped", False)
        and not has_forward
        and not module._forward_pre_hooks
        and not module._forward_hooks
        and not module._backward_pre_hooks
        and not module._backward_hooks
    )


_standard_forward_wrapper = _port_wrapper(
    _standard_forward,
    "openwam.model.kernels.fused_blocks._standard_forward",
    "openwam.compute",
)


def _wan_forward_supported(block) -> bool:
    from openwam.model.video_backbone.wan.models.dit import (
        AttentionModule,
        CrossAttention,
        DiTBlock,
        GateModule,
        RMSNorm,
        SelfAttention,
    )
    from turbo_physai.optimizations.models.openwam.norms import LayerNorm

    return (
        _standard_forward(block, DiTBlock)
        and all(
            _standard_forward(getattr(block, name), (torch.nn.LayerNorm, LayerNorm))
            for name in ("norm1", "norm2", "norm3")
        )
        and _attention_forward_supported(
            block.self_attn, SelfAttention, RMSNorm, AttentionModule
        )
        and _attention_forward_supported(
            block.cross_attn, CrossAttention, RMSNorm, AttentionModule
        )
        and _ffn_forward_supported(block.ffn)
        and _standard_forward(block.gate, GateModule)
    )


_wan_forward_supported_wrapper = _port_wrapper(
    _wan_forward_supported,
    "openwam.model.kernels.fused_blocks._wan_forward_supported",
    "openwam.compute",
)
