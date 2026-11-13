# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3-VL text decoder 逐层 torch.compile"""

from __future__ import annotations

import functools
import types
from typing import Any

import torch

COMPILE_OPTIONS = {"dynamic": True}


def _kqv_impl(self, hidden_states):
    """Qwen3VLTextDecoderLayer.forward 的 compute_kqv 分支。"""
    param_dtype = self.self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)

    hidden_states = self.input_layernorm(hidden_states)
    hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

    query_state = self.self_attn.q_norm(self.self_attn.q_proj(hidden_states).view(hidden_shape))
    key_state = self.self_attn.k_norm(self.self_attn.k_proj(hidden_states).view(hidden_shape))
    value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
    return query_state, key_state, value_state


def _oa_impl(self, hidden_states, att_output, start, end):
    """Qwen3VLTextDecoderLayer.forward 的 output_atten 分支。"""
    param_dtype = self.self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)
    if att_output is not None:
        att_output = att_output.to(param_dtype)

    if att_output.dtype != self.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
    out_emb = self.self_attn.o_proj(att_output[:, start:end])

    out_emb += hidden_states
    after_first_residual = out_emb
    out_emb = self.post_attention_layernorm(out_emb)
    out_emb = self.mlp(out_emb)
    out_emb += after_first_residual

    return out_emb


def _stabilise(tensor, param_dtype):
    """转成权重 dtype，view 则物化一份，让 forward 和重算走同一张图。"""
    converted = tensor.to(param_dtype)
    if converted is tensor and getattr(tensor, "_base", None) is not None:
        converted = tensor.clone()
    if not converted.is_contiguous():
        converted = converted.contiguous()
    return converted


def _compile_layer_forward(layer) -> Any:
    """给一层的两个路径各编一张图，把 hybrid 挂到实例的 forward 上。"""
    if torch.compiler.is_compiling():
        return layer.forward
    hybrid = layer.__dict__.get("_tp_compiled_forward")
    if hybrid is None:
        eager_forward = layer.forward
        compiled_kqv = torch.compile(
            types.MethodType(_kqv_impl, layer), **COMPILE_OPTIONS
        )
        compiled_oa = torch.compile(
            types.MethodType(_oa_impl, layer), **COMPILE_OPTIONS
        )
        param_dtype = layer.self_attn.q_proj.weight.dtype

        def hybrid(*args, **kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]

            compute_kqv = kwargs.get("compute_kqv", False)
            if not compute_kqv and len(args) > 4:
                compute_kqv = args[4]
            if compute_kqv:
                att_output = args[1] if len(args) > 1 else kwargs.get("att_output")
                if att_output is None:
                    return compiled_kqv(_stabilise(hidden_states, param_dtype))
                return eager_forward(*args, **kwargs)

            output_atten = kwargs.get("output_atten", False)
            if not output_atten and len(args) > 5:
                output_atten = args[5]
            if output_atten:
                att_output = args[1] if len(args) > 1 else kwargs.get("att_output")
                if att_output is not None:
                    start = args[2] if len(args) > 2 else kwargs.get("start", 0)
                    end = args[3] if len(args) > 3 else kwargs.get("end", 0)
                    return compiled_oa(
                        _stabilise(hidden_states, param_dtype),
                        _stabilise(att_output, param_dtype),
                        start,
                        end,
                    )

            return eager_forward(*args, **kwargs)

        layer.__dict__["_tp_compiled_forward"] = hybrid
        layer.__dict__["forward"] = hybrid
    return hybrid


def _text_layers(model) -> list:
    return list(model.qwenvl.model.language_model.layers)


def _ensure_compiled(model) -> None:
    """每个模型实例只装一次，trace 期间直接返回。"""
    if torch.compiler.is_compiling():
        return
    if model.__dict__.get("_tp_text_layers_compiled", False):
        return
    for layer in _text_layers(model):
        _compile_layer_forward(layer)
    model.__dict__["_tp_text_layers_compiled"] = True


def install_on_init(original, _options=None):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _ensure_compiled(self)
        return None

    return wrapped


def compile_text_layers(
    self,
    attention_mask=None,
    position_ids=None,
    vlm_position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    fill_kv_cache=None,
    ada_cond=None,
    visual_pos_masks=None,
    deepstack_visual_embeds=None,
):
    _ensure_compiled(self)
    from .decoder_memory import forward

    return forward(
        self,
        attention_mask,
        position_ids,
        vlm_position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
    )
