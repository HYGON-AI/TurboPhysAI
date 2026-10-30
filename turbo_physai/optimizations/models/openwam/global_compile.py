# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from turbo_physai.optimizations.models.openwam.settings import value

"""Compile training blocks, with explicit boundaries for optional eager kernels."""

import logging
from typing import Optional

import torch
from einops import rearrange
from torch import Tensor

from turbo_physai.optimizations.models.openwam.ports.action import (
    _modulate,
    rope_apply_1d,
)
from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply
from turbo_physai.optimizations.models.openwam.settings import enabled
from turbo_physai.optimizations.models.openwam.attention import (
    separate_attention_plan,
    split_attention_separate,
)
from turbo_physai.optimizations.models.openwam.linear_bias2d import (
    cross_attention_kwargs,
    video_ffn,
    video_linear,
)
from turbo_physai.optimizations.models.openwam.sac import (
    active_ffn_boundary,
    dispatcher,
    run_attention,
    run_ffn,
    run_norm,
    run_projection,
)

logger = logging.getLogger(__name__)


def compile_scope():
    if not enabled("global_compile"):
        return set()
    scope = value("compile_scope", "all").strip().lower()
    if scope not in ("all", "mot", "vae"):
        raise ValueError("compile_scope must be all, mot or vae")
    return {"mot", "vae"} if scope == "all" else {scope}


def _options():
    import torch._inductor

    # Retain BF16 rounding at the original expression boundaries.
    return (
        {"emulate_precision_casts": True}
        if "emulate_precision_casts" in torch._inductor.list_options()
        else {}
    )


def mark_mot_text_dynamic(vstate, astate):
    """Annotate prepared text inputs before entering the compiled MoT layer."""
    if not enabled("text_trim"):
        return
    from torch._dynamo import mark_dynamic

    payload = astate.payload
    tensors = (
        (vstate.context, 1),
        (vstate.context_mask, -1),
        (payload.context, 1),
        # Action masks can be [B, L], [B, S_action, L] or [B, H, S_action, L].
        (payload.context_mask, -1),
    )
    for tensor, dim in tensors:
        if tensor is None:
            continue
        axis = dim if dim >= 0 else tensor.ndim + dim
        # PyTorch specializes sizes 0/1; leave those as separate static cases.
        if tensor.shape[axis] <= 1 or axis in getattr(
            tensor, "_dynamo_dynamic_indices", ()
        ):
            continue
        mark_dynamic(tensor, axis)


def _stage1_pre_attn(
    video_block,
    action_block,
    v_hidden_states: Tensor,
    v_time_mod: Tensor,
    v_rope_freqs: Tensor,
    a_x: Tensor,
    a_t_mod: Tensor,
    a_freqs: Tensor,
    a_num_heads: int,
    separate_qkv: bool = False,
):
    has_seq = v_time_mod.dim() == 4
    chunk_dim = 2 if has_seq else 1
    v_chunks = (
        video_block.modulation.to(dtype=v_time_mod.dtype, device=v_time_mod.device)
        + v_time_mod
    ).chunk(6, dim=chunk_dim)
    if has_seq:
        v_chunks = tuple(c.squeeze(2) for c in v_chunks)
    v_shift_msa, v_scale_msa, v_gate_msa, v_shift_mlp, v_scale_mlp, v_gate_mlp = (
        v_chunks
    )

    v_attn_input = modulate(
        run_norm(video_block.norm1, v_hidden_states), v_shift_msa, v_scale_msa
    )
    v_sa = video_block.self_attn
    q_v = run_norm(v_sa.norm_q, video_linear(v_sa.q, v_attn_input))
    k_v = run_norm(v_sa.norm_k, video_linear(v_sa.k, v_attn_input))
    v_v = video_linear(v_sa.v, v_attn_input)
    q_v = rope_apply(q_v, v_rope_freqs, v_sa.num_heads)
    k_v = rope_apply(k_v, v_rope_freqs, v_sa.num_heads)

    a_chunks = (
        action_block.modulation.to(dtype=a_t_mod.dtype, device=a_t_mod.device) + a_t_mod
    ).chunk(6, dim=1)
    a_shift_msa, a_scale_msa, a_gate_msa, a_shift_mlp, a_scale_mlp, a_gate_mlp = (
        a_chunks
    )

    a_attn_input = _modulate(
        run_norm(action_block.self_attn_norm, a_x), a_shift_msa, a_scale_msa
    )
    a_sa = action_block.self_attn
    q_a = run_norm(a_sa.norm_q, run_projection(a_sa.q, a_attn_input))
    k_a = run_norm(a_sa.norm_k, run_projection(a_sa.k, a_attn_input))
    v_a = run_projection(a_sa.v, a_attn_input)

    q_a = rearrange(q_a, "b s (n d) -> b n s d", n=a_num_heads)
    k_a = rearrange(k_a, "b s (n d) -> b n s d", n=a_num_heads)
    q_a = rope_apply_1d(q_a, a_freqs)
    k_a = rope_apply_1d(k_a, a_freqs)
    q_a = rearrange(q_a, "b n s d -> b s (n d)", n=a_num_heads)
    k_a = rearrange(k_a, "b n s d -> b s (n d)", n=a_num_heads)

    if separate_qkv:
        # Keep one packed K/V allocation, just as in the original path. Returning
        # raw video K/V as well as an eager joint cat pins two extra video-sized
        # buffers per layer for attention backward (about 2 GiB over 30 layers).
        attention_inputs = (
            q_v,
            torch.cat([k_v, k_a], dim=1),
            torch.cat([v_v, v_a], dim=1),
            q_a,
            None,
            None,
        )
    else:
        attention_inputs = (
            torch.cat([q_v, q_a], dim=1),
            torch.cat([k_v, k_a], dim=1),
            torch.cat([v_v, v_a], dim=1),
            None,
            None,
            None,
        )
    return (
        *attention_inputs,
        v_gate_msa,
        v_shift_mlp,
        v_scale_mlp,
        v_gate_mlp,
        a_gate_msa,
        a_shift_mlp,
        a_scale_mlp,
        a_gate_mlp,
    )


def _stage2_joint_pre_cross_attn(mixed, s_video, s_action, *args):
    # Keep splitting inside the compiled graph on the original attention path.
    attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
    return _stage2_pre_cross_attn(attn_v, attn_a, *args)


def _stage2_pre_cross_attn(
    attn_v: Tensor,
    attn_a: Tensor,
    video_block,
    action_block,
    v_residual_x: Tensor,
    v_gate_msa: Tensor,
    a_residual_x: Tensor,
    a_gate_msa: Tensor,
    v_context: Optional[Tensor],
    v_context_mask: Optional[Tensor],
    a_context: Optional[Tensor],
    a_context_mask: Optional[Tensor],
    linear_bias2d: bool,
):
    attn_v = attn_v.contiguous()
    attn_a = attn_a.contiguous()

    # Video post-self-attn & cross-attn QKV prep
    v_hidden_states = video_block.gate(
        v_residual_x, v_gate_msa, video_linear(video_block.self_attn.o, attn_v)
    )

    ca_v = video_block.cross_attn
    if ca_v.has_image_input:
        v_img = v_context[:, :257]
        v_ctx = v_context[:, 257:]
        v_ctx_mask = v_context_mask[..., 257:] if v_context_mask is not None else None
    else:
        v_ctx = v_context
        v_ctx_mask = v_context_mask

    if v_ctx_mask is not None:
        if v_ctx_mask.dim() == 2:
            v_ctx_mask = (
                v_ctx_mask.unsqueeze(1)
                .expand(-1, v_hidden_states.shape[1], -1)
                .unsqueeze(1)
            )
        elif v_ctx_mask.dim() == 3:
            v_ctx_mask = v_ctx_mask.unsqueeze(1)

    v_norm3 = run_norm(video_block.norm3, v_hidden_states)
    v_ca_q = run_norm(
        ca_v.norm_q,
        video_linear(ca_v.q, v_norm3)
        if linear_bias2d
        else run_projection(ca_v.q, v_norm3),
    )
    v_ca_k = run_norm(ca_v.norm_k, run_projection(ca_v.k, v_ctx))
    v_ca_v = run_projection(ca_v.v, v_ctx)

    if ca_v.has_image_input:
        v_ca_k_img = run_norm(ca_v.norm_k_img, run_projection(ca_v.k_img, v_img))
        v_ca_v_img = run_projection(ca_v.v_img, v_img)
    else:
        v_ca_k_img = None
        v_ca_v_img = None

    # Action post-self-attn & cross-attn QKV prep
    a_x = action_block.gate(
        a_residual_x, a_gate_msa, run_projection(action_block.self_attn.o, attn_a)
    )

    if a_context is not None:
        ca_a = action_block.cross_attn
        a_text_mask = a_context_mask
        if a_text_mask is not None:
            if a_text_mask.dim() == 2:
                a_text_mask = a_text_mask.unsqueeze(1).expand(-1, a_x.shape[1], -1)
            elif a_text_mask.dim() not in (3, 4):
                raise ValueError(
                    "ActionDiTState.context_mask must be [B, L], [B, T_action, L], "
                    f"or broadcastable [B, heads, T_action, L], got {tuple(a_text_mask.shape)}"
                )
        a_normed = run_norm(action_block.context_attn_norm, a_x)
        a_ca_q = run_norm(ca_a.norm_q, run_projection(ca_a.q, a_normed))
        a_ca_k = run_norm(ca_a.norm_k, run_projection(ca_a.k, a_context))
        a_ca_v = run_projection(ca_a.v, a_context)
    else:
        a_text_mask = None
        a_ca_q = None
        a_ca_k = None
        a_ca_v = None

    return (
        v_hidden_states,
        a_x,
        v_ca_q,
        v_ca_k,
        v_ca_v,
        v_ctx_mask,
        v_ca_k_img,
        v_ca_v_img,
        a_ca_q,
        a_ca_k,
        a_ca_v,
        a_text_mask,
    )


def _stage3_pre_ffn(
    video_block,
    action_block,
    v_hidden_states: Tensor,
    v_attn_out: Tensor,
    v_shift_mlp: Tensor,
    v_scale_mlp: Tensor,
    a_x: Tensor,
    a_attn_out: Optional[Tensor],
    a_shift_mlp: Tensor,
    a_scale_mlp: Tensor,
    a_gate_mlp: Tensor,
    linear_bias2d: bool,
    sac_active_ffn: bool,
):
    ca_v = video_block.cross_attn
    v_cross_out = (
        video_linear(ca_v.o, v_attn_out)
        if linear_bias2d
        else run_projection(ca_v.o, v_attn_out)
    )
    v_hidden_states = v_hidden_states + v_cross_out
    v_mlp_input = modulate(
        run_norm(video_block.norm2, v_hidden_states), v_shift_mlp, v_scale_mlp
    )

    if a_attn_out is not None:
        ca_a = action_block.cross_attn
        a_cross_out = run_projection(ca_a.o, a_attn_out)
        a_x = a_x + a_cross_out

    a_mlp_input = _modulate(
        run_norm(action_block.ffn_norm, a_x), a_shift_mlp, a_scale_mlp
    )

    if not sac_active_ffn:
        a_mlp_output = action_block.ffn(a_mlp_input)
        a_x = action_block.gate(a_x, a_gate_mlp, a_mlp_output)
        return v_hidden_states, v_mlp_input, a_x, None

    return v_hidden_states, v_mlp_input, a_x, a_mlp_input


def _stage4_post_ffn(
    video_block,
    action_block,
    v_hidden_states: Tensor,
    v_gate_mlp: Tensor,
    v_mlp_output: Tensor,
    a_x: Optional[Tensor],
    a_gate_mlp: Optional[Tensor],
    a_mlp_output: Optional[Tensor],
):
    v_hidden_states = video_block.gate(v_hidden_states, v_gate_mlp, v_mlp_output)
    if a_mlp_output is not None:
        a_x = action_block.gate(a_x, a_gate_mlp, a_mlp_output)
    return v_hidden_states, a_x


def compile_mot_layer(driver):
    from turbo_physai.optimizations.models.openwam.sac import attention_boundary

    driver._compiled_attention = (
        torch.compiler.disable(driver._mixed_attention)
        if enabled("mot_split_attn") or attention_boundary()
        else driver._mixed_attention
    )

    compiled_stage1 = torch.compile(_stage1_pre_attn, dynamic=False, options=_options())
    compiled_stage2 = torch.compile(
        _stage2_pre_cross_attn, dynamic=False, options=_options()
    )
    compiled_stage2_joint = torch.compile(
        _stage2_joint_pre_cross_attn, dynamic=False, options=_options()
    )
    compiled_stage3 = torch.compile(_stage3_pre_ffn, dynamic=False, options=_options())
    compiled_stage4 = torch.compile(_stage4_post_ffn, dynamic=False, options=_options())
    a_num_heads = int(driver.ab.num_heads)

    @dispatcher
    def layer(video_block, action_block, vstate, astate, attn_mask):
        v_residual_x = vstate.hidden_states
        s_video = v_residual_x.shape[1]
        payload = astate.payload
        a_residual_x = payload.x_action
        s_action = a_residual_x.shape[1]
        plan = separate_attention_plan(attn_mask, s_video, s_action)

        # Stage 1: Pre-attention projection & concatenation
        (
            q_v,
            k_v,
            v_v,
            q_a,
            k_a,
            v_a,
            v_gate_msa,
            v_shift_mlp,
            v_scale_mlp,
            v_gate_mlp,
            a_gate_msa,
            a_shift_mlp,
            a_scale_mlp,
            a_gate_mlp,
        ) = compiled_stage1(
            video_block,
            action_block,
            v_residual_x,
            vstate.time_mod,
            vstate.rope_freqs,
            a_residual_x,
            payload.t_mod,
            payload.action_freqs,
            a_num_heads,
            plan is not None,
        )

        # Eager Joint Mixed Attention
        if plan is None:
            mixed = driver._mixed_attention(q_v, k_v, v_v, attn_mask)
            stage2 = compiled_stage2_joint
            stage2_inputs = (mixed, s_video, s_action)
        else:
            from openwam.model.video_backbone.wan.models.dit import flash_attention

            attn_v, attn_a = run_attention(
                split_attention_separate,
                q_v,
                None,
                None,
                q_a,
                None,
                None,
                plan,
                a_num_heads,
                flash_attention,
                joint_kv=(k_v, v_v),
                label="mixed_attention",
            )
            stage2 = compiled_stage2
            stage2_inputs = (attn_v, attn_a)

        # Stage 2: Post-self-attn gate & cross-attn QKV preparation
        linear_bias2d = bool(
            cross_attention_kwargs(video_block.cross_attn).get("linear_bias2d", False)
        )
        (
            v_hidden_states,
            a_x,
            v_ca_q,
            v_ca_k,
            v_ca_v,
            v_ctx_mask,
            v_ca_k_img,
            v_ca_v_img,
            a_ca_q,
            a_ca_k,
            a_ca_v,
            a_text_mask,
        ) = stage2(
            *stage2_inputs,
            video_block,
            action_block,
            v_residual_x,
            v_gate_msa,
            a_residual_x,
            a_gate_msa,
            vstate.context,
            vstate.context_mask,
            payload.context,
            payload.context_mask,
            linear_bias2d,
        )

        # Eager Cross-Attention Kernels
        ca_v = video_block.cross_attn
        v_attn_out = run_attention(
            ca_v.attn,
            v_ca_q,
            v_ca_k,
            v_ca_v,
            attn_mask=v_ctx_mask,
            label="video_cross_attention",
        )
        if ca_v.has_image_input:
            from openwam.model.video_backbone.wan.models.dit import flash_attention

            v_img_out = run_attention(
                flash_attention,
                v_ca_q,
                v_ca_k_img,
                v_ca_v_img,
                num_heads=ca_v.num_heads,
                label="video_cross_attention",
            )
            v_attn_out = v_attn_out + v_img_out

        if a_ca_q is not None:
            ca_a = action_block.cross_attn
            a_attn_out = run_attention(
                ca_a._attend,
                a_ca_q,
                a_ca_k,
                a_ca_v,
                a_text_mask,
                label="action_cross_attention",
            )
        else:
            a_attn_out = None

        # Stage 3: Cross-attn output projection + norm2 + modulate (with dynamic non-SAC Action FFN fusion)
        sac_active_ffn = active_ffn_boundary()
        v_hidden_states, v_mlp_input, a_x, a_mlp_input = compiled_stage3(
            video_block,
            action_block,
            v_hidden_states,
            v_attn_out,
            v_shift_mlp,
            v_scale_mlp,
            a_x,
            a_attn_out,
            a_shift_mlp,
            a_scale_mlp,
            a_gate_mlp,
            linear_bias2d,
            sac_active_ffn,
        )

        if sac_active_ffn:
            v_mlp_output = run_ffn(video_block.ffn, v_mlp_input, video=True)
            a_mlp_output = run_ffn(action_block.ffn, a_mlp_input)
            v_hidden_states, a_x = compiled_stage4(
                video_block,
                action_block,
                v_hidden_states,
                v_gate_mlp,
                v_mlp_output,
                a_x,
                a_gate_mlp,
                a_mlp_output,
            )
        else:
            v_mlp_output = video_ffn(video_block.ffn, v_mlp_input)
            v_hidden_states, _ = compiled_stage4(
                video_block,
                action_block,
                v_hidden_states,
                v_gate_mlp,
                v_mlp_output,
                None,
                None,
                None,
            )
        vstate.hidden_states = v_hidden_states
        payload.x_action = a_x
        return vstate, astate

    return layer


def compile_vae_encoder(vae):
    encoder = vae.model.encoder
    # Replace the callable, rather than the module, to retain checkpoint keys,
    # type checks and the causal feature-cache protocol.
    encoder.forward = torch.compile(
        encoder.forward, dynamic=False, fullgraph=True, options=_options()
    )
