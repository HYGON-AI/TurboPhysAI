# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import logging

import torch
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)

_GC_KWARGS = {"use_reentrant": False}


def _align(value, dtype):
    if torch.is_tensor(value):
        return value if value.dtype == dtype else value.to(dtype)
    if isinstance(value, (list, tuple)):
        return type(value)(_align(item, dtype) for item in value)
    return value


def _checkpointed(layer, args, kwargs):
    dtype = next(layer.parameters()).dtype
    return checkpoint(
        layer,
        *(_align(arg, dtype) for arg in args),
        use_reentrant=False,
        **{name: _align(value, dtype) for name, value in kwargs.items()},
    )


def _enable_gc(module, name: str) -> bool:
    if module is None or not hasattr(module, "gradient_checkpointing_enable"):
        return False
    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(_GC_KWARGS))
    module.train(module.training)
    logger.info("lingbot-vla: gradient checkpointing enabled on %s", name)
    return True


def _qwenvl_with_expert(out):
    inner = getattr(out, "model", out)
    wrapper = getattr(inner, "qwenvl_with_expert", None)
    if wrapper is None:
        logger.warning("lingbot-vla: qwenvl_with_expert not found, skip GC patch")
    return wrapper


def wrap_enable_gc(original, options=None):
    @functools.wraps(original)
    def wrapped(model, *args, **kwargs):
        out = original(model, *args, **kwargs)
        if not kwargs.get("enable_gradient_checkpointing", True):
            return out
        wrapper = _qwenvl_with_expert(out)
        if wrapper is not None:
            hit = False
            for attr in ("qwenvl", "qwen_expert"):
                hit |= _enable_gc(getattr(wrapper, attr, None), attr)
            if hit:
                wrapper.gradient_checkpointing = True
        return out

    return wrapped


def wrap_llm_gc(original, options=None):
    options = options or {}
    num_layers = options.get("num_layers")
    vl_layers = options.get("vl_layers", 0)
    expert_layers = options.get("expert_layers", 0)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.gc_layers = num_layers
        self.gc_vl_layers = vl_layers
        self.gc_expert_layers = expert_layers

    return wrapped


def _checkpoint_layer(layer_idx, is_expert, gc_layers, vl_layers, expert_layers):
    """num_layers 控制两个模型共同重算的层数；其为 0 时按 vl/expert 分别控制。"""
    if gc_layers is None:
        return True
    if gc_layers:
        return layer_idx < gc_layers
    return layer_idx < (expert_layers if is_expert else vl_layers)


def ada_rms_norm_forward(self, hidden_states, cond):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

    hidden_states = self.weight * hidden_states
    # cond 固定是 fp32；enable_fp32: false 时专家权重是 bf16，这里对齐到权重 dtype。
    gamma = self.gamma(cond.to(self.gamma.weight.dtype)).unsqueeze(1)
    beta = self.beta(cond.to(self.beta.weight.dtype)).unsqueeze(1)
    hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
    return hidden_states.to(input_dtype)


def forward(
    self,
    attention_mask=None,
    position_ids=None,
    vlm_position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    fill_kv_cache=None,
    ada_cond=None,
    norm_qkv=False,
):
    from lingbotvla.models.vla.pi0.utils import apply_rope

    models = [self.qwenvl.model, self.qwen_expert.model]
    use_gc = self.training and getattr(self, "gradient_checkpointing", False)
    gc_layers = getattr(self, "gc_layers", None)
    gc_vl_layers = getattr(self, "gc_vl_layers", 0)
    gc_expert_layers = getattr(self, "gc_expert_layers", 0)

    num_layers = self.qwenvl.config.num_hidden_layers
    for layer_idx in range(num_layers):
        query_states = []
        key_states = []
        value_states = []
        gates = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            layer = models[i].layers[layer_idx]
            if i == 1:
                query_state, key_state, value_state, gate = layer(
                    hidden_states, compute_kqv=True, ada_cond=ada_cond, norm_qkv=norm_qkv
                )
            else:
                query_state, key_state, value_state = layer(
                    hidden_states, compute_kqv=True, norm_qkv=norm_qkv
                )
                gate = None

            if query_state.dtype != torch.float32:
                query_state = query_state.to(torch.float32)
                key_state = key_state.to(torch.float32)
                value_state = value_state.to(torch.float32)
            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)
            gates.append(gate)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)

        key_states, value_states, past_key_values = self.handle_kv_cache(
            key_states,
            value_states,
            layer_idx,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )

        att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            layer = models[i].layers[layer_idx]
            extra = (
                {"ada_cond": ada_cond, "gate": gates[0] if len(gates) == 1 else gates[i]}
                if i == 1
                else {}
            )
            if use_gc and _checkpoint_layer(
                layer_idx, i == 1, gc_layers, gc_vl_layers, gc_expert_layers
            ):
                out_emb = _checkpointed(
                    layer,
                    (hidden_states, att_output, start, end),
                    {"output_atten": True, **extra},
                )
            else:
                out_emb = layer(hidden_states, att_output, start, end, output_atten=True, **extra)
            outputs_embeds.append(out_emb)
            start = end

        inputs_embeds = outputs_embeds

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is None:
            outputs_embeds.append(None)
            continue
        if self.config.final_norm_adanorm:
            if i == 1:
                out_emb, _ = models[i].norm(hidden_states, ada_cond)
            else:
                out_emb = models[i].norm(hidden_states)
        else:
            out_emb = models[i].norm(hidden_states)
        outputs_embeds.append(out_emb)

    return outputs_embeds, past_key_values
