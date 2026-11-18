# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen2 action expert（MoE decoder）的分支级 torch.compile。"""

from __future__ import annotations

import functools
import os
import types
from typing import Any

import torch

COMPILE_OPTIONS = {"dynamic": True}


def _kqv_impl(self, hidden_states, ada_cond=None):
    param_dtype = self.self_attn.q_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)
    if ada_cond is not None:
        ada_cond = ada_cond.to(param_dtype)

    if ada_cond is not None:
        hidden_states = self.input_layernorm(hidden_states, ada_cond)
    else:
        hidden_states = self.input_layernorm(hidden_states)
    hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

    query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
    key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
    value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
    return query_state, key_state, value_state


def _oa_dense_impl(self, hidden_states, att_output, start, end, ada_cond=None):
    """output_atten 分支的 dense 段：o_proj + 第一个残差 + post_attention_layernorm。

    MoE不在图里，返回 norm 后的输入和残差，由 python 侧接上。
    """
    param_dtype = self.self_attn.o_proj.weight.dtype
    hidden_states = hidden_states.to(param_dtype)
    att_output = att_output.to(param_dtype)
    if ada_cond is not None:
        ada_cond = ada_cond.to(param_dtype)

    out_emb = self.self_attn.o_proj(att_output[:, start:end])
    out_emb = out_emb + hidden_states
    residual = out_emb
    if ada_cond is not None:
        normed = self.post_attention_layernorm(out_emb, ada_cond)
    else:
        normed = self.post_attention_layernorm(out_emb)
    return normed, residual


def _stabilise(tensor, param_dtype):
    if tensor is None:
        return None
    converted = tensor.to(param_dtype)
    if converted is tensor and getattr(tensor, "_base", None) is not None:
        converted = tensor.clone()
    if not converted.is_contiguous():
        converted = converted.contiguous()
    return converted


def _compile_layer_forward(layer) -> Any:
    if torch.compiler.is_compiling():
        return layer.forward
    hybrid = layer.__dict__.get("_tp_compiled_forward")
    if hybrid is None:
        eager_forward = layer.forward
        compiled_kqv = torch.compile(
            types.MethodType(_kqv_impl, layer), **COMPILE_OPTIONS
        )
        compiled_oa = torch.compile(
            types.MethodType(_oa_dense_impl, layer), **COMPILE_OPTIONS
        )

        def hybrid(*args, **kwargs):
            compute_kqv = kwargs.get("compute_kqv", False)
            output_atten = kwargs.get("output_atten", False)
            if not compute_kqv and not output_atten:
                compute_kqv = bool(len(args) > 4 and args[4])
                output_atten = bool(len(args) > 5 and args[5])
            hidden_states = args[0] if args else kwargs["hidden_states"]
            ada_cond = kwargs.get("ada_cond")
            if ada_cond is None and len(args) > 6:
                ada_cond = args[6]

            param_dtype = layer.self_attn.q_proj.weight.dtype
            stable_cond = _stabilise(ada_cond, param_dtype)

            if compute_kqv and not output_atten:
                return compiled_kqv(
                    _stabilise(hidden_states, param_dtype), ada_cond=stable_cond
                )

            if output_atten:
                att_output = args[1] if len(args) > 1 else kwargs.get("att_output")
                if att_output is not None:
                    start = args[2] if len(args) > 2 else kwargs.get("start", 0)
                    end = args[3] if len(args) > 3 else kwargs.get("end", 0)
                    normed, residual = compiled_oa(
                        _stabilise(hidden_states, param_dtype),
                        _stabilise(att_output, param_dtype),
                        start,
                        end,
                        ada_cond=stable_cond,
                    )
                    mlp_out = layer.mlp(normed)
                    router_logits = None
                    if isinstance(mlp_out, tuple):
                        mlp_out, router_logits = mlp_out
                    return mlp_out + residual, router_logits

            return eager_forward(*args, **kwargs)

        layer.__dict__["_tp_compiled_forward"] = hybrid
        layer.__dict__["forward"] = hybrid
    return hybrid


def _expert_layers(model) -> list:
    return list(model.model.layers)


def _ensure_compiled(model) -> None:
    if torch.compiler.is_compiling():
        return
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return
    if model.__dict__.get("_tp_expert_layers_compiled", False):
        return
    for layer in _expert_layers(model):
        _compile_layer_forward(layer)
    model.__dict__["_tp_expert_layers_compiled"] = True


def install_on_init(original, _options=None):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _ensure_compiled(self)
        return None

    return wrapped




def deepstack_process(
    self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
) -> torch.Tensor:
    visual_pos_masks = visual_pos_masks.to(hidden_states.device)
    visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    out = hidden_states.clone()
    out[visual_pos_masks, :] = local_this
    return out
