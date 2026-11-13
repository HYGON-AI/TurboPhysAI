# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
Qwen2 action expert（MoE decoder）的逐层 torch.compile。
"""

from __future__ import annotations

import functools
import types
from typing import Any

import torch

COMPILE_OPTIONS = {"dynamic": True}


def _kqv_impl(self, hidden_states, ada_cond=None):
    """
    Qwen2DecoderLayer.forward 的 compute_kqv 分支。
    """
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


def _stabilise(tensor, param_dtype):
    """
    转成权重 dtype，只在必要时物化布局，在 checkpoint 之前调用。
    """
    if tensor is None:
        return None
    converted = tensor.to(param_dtype)
    if converted is tensor and getattr(tensor, "_base", None) is not None:
        converted = tensor.clone()
    if not converted.is_contiguous():
        converted = converted.contiguous()
    return converted


def _compile_layer_forward(layer) -> Any:
    """给一层编 compute_kqv 一张图，把 hybrid 挂到实例 forward 上。"""
    if torch.compiler.is_compiling():
        return layer.forward
    hybrid = layer.__dict__.get("_tp_compiled_forward")
    if hybrid is None:
        eager_forward = layer.forward
        bound_kqv = types.MethodType(_kqv_impl, layer)
        compiled_kqv = torch.compile(bound_kqv, **COMPILE_OPTIONS)
        param_dtype = layer.self_attn.q_proj.weight.dtype

        def hybrid(*args, **kwargs):
            if kwargs.get("output_atten", False) or (len(args) > 5 and args[5]):
                return eager_forward(*args, **kwargs)
            hidden_states = args[0] if args else kwargs["hidden_states"]
            if hidden_states.dtype is not param_dtype:
                return eager_forward(*args, **kwargs)
            if not hidden_states.is_contiguous():
                hidden_states = hidden_states.contiguous()
            ada_cond = kwargs.get("ada_cond")
            if ada_cond is None and len(args) > 6:
                ada_cond = args[6]
            return compiled_kqv(hidden_states, ada_cond=ada_cond)

        layer.__dict__["_tp_compiled_forward"] = hybrid
        layer.__dict__["forward"] = hybrid
    return hybrid


def install_call_guard(original, _options=None):
    @functools.wraps(original)
    def guarded_call(self, *args, **kwargs):
        param_dtype = self.self_attn.q_proj.weight.dtype
        if args:
            args = list(args)
            for index in (0, 1):
                if index < len(args) and torch.is_tensor(args[index]):
                    args[index] = _stabilise(args[index], param_dtype)
            args = tuple(args)
        ada_cond = kwargs.get("ada_cond")
        if torch.is_tensor(ada_cond):
            kwargs["ada_cond"] = _stabilise(ada_cond, param_dtype)
        return original(self, *args, **kwargs)

    return guarded_call


def _expert_layers(model) -> list:
    return list(model.model.layers)


def _ensure_compiled(model) -> None:
    if torch.compiler.is_compiling():
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
