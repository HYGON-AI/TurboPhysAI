# Portions are derived from OpenPI:
# https://github.com/Physical-Intelligence/openpi
# File: src/openpi/models_pytorch/gemma_pytorch.py
# Commit: 215abfb217dbac7d5f1273282331b9b1866c0479
# Upstream license: Apache-2.0
#
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

"""Patch-equivalent selective Gemma gradient-checkpoint replacement."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import torch
from transformers.models.gemma import modeling_gemma


_CHECKPOINT_ONLY_KWARGS = {
    "use_reentrant",
    "preserve_rng_state",
    "context_fn",
    "determinism_check",
    "debug",
    "early_stop",
}




def wrap_gemma_forward(original, options: Mapping[str, Any]):
    """Replace Gemma's forward loop with the patch's stride-aware loop.

    The official checkout currently calls ``checkpoint`` for every layer.  A
    wrapper around that helper is not equivalent under an outer
    ``torch.compile``: Dynamo traces the wrapper and creates a graph variant
    for each Python layer index.  Owning the forward loop here keeps the
    stride-zero path as direct calls, exactly as the patched source does,
    while leaving the official checkout untouched.
    """
    stride = int(options.get("layer_stride", 0))
    if stride < 0:
        raise ValueError("layer_stride must be non-negative")
    if os.getenv("RANK", "0") == "0":
        state = "disabled" if stride == 0 else f"every {stride} layer(s)"
        print(
            f"[TurboPhysAI] Gemma joint forward: checkpoint {state} (layer_stride={stride}); "
            "active only in training when checkpointing is enabled; "
            "other forward paths use the original implementation",
            flush=True,
        )

    def wrapped(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        # Keep the two inference paths identical to the reference patch.  The
        # training path below is the only path on which the stride switch is
        # relevant; delegating these paths also keeps cache/past-key behavior
        # owned by the official implementation.
        if inputs_embeds is None or inputs_embeds[1] is None:
            return original(
                self,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )

        if inputs_embeds[0] is None:
            return original(
                self,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )

        if adarms_cond is None:
            adarms_cond = [None, None]

        models = [self.paligemma.language_model, self.gemma_expert.model]
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
            if not self.gemma_expert.model.gradient_checkpointing:
                print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                self.gemma_expert.model.gradient_checkpointing = True
            use_gradient_checkpointing = True

        if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
            print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
            print(f"Model training mode: {self.training}")
            print(
                "Gemma expert model has gradient_checkpointing attr: "
                f"{hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
            )
            if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                print(
                    "Gemma expert model gradient_checkpointing value: "
                    f"{self.gemma_expert.model.gradient_checkpointing}"
                )
            self._debug_gc_printed = True

        def compute_layer_complete(layer_idx, layer_inputs, layer_attention_mask, layer_position_ids, layer_adarms_cond):
            # Keep this list inside the nested function, as in the reference
            # implementation.  Capturing it in the outer compiled wrapper
            # creates a different Dynamo closure/guard topology.
            models = [self.paligemma.language_model, self.gemma_expert.model]
            query_states = []
            key_states = []
            value_states = []
            gates = []
            for i, hidden_states in enumerate(layer_inputs):
                layer = models[i].layers[layer_idx]
                hidden_states, gate = layer.input_layernorm(
                    hidden_states, cond=layer_adarms_cond[i]
                )
                gates.append(gate)
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
                key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
                value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

            query_states = torch.cat(query_states, dim=2)
            key_states = torch.cat(key_states, dim=2)
            value_states = torch.cat(value_states, dim=2)
            dummy_tensor = torch.zeros(
                query_states.shape[0],
                query_states.shape[2],
                query_states.shape[-1],
                device=query_states.device,
                dtype=query_states.dtype,
            )
            cos, sin = self.paligemma.model.language_model.rotary_emb(
                dummy_tensor, layer_position_ids
            )
            query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, unsqueeze_dim=1
            )
            batch_size = query_states.shape[0]
            scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling
            att_output, _ = modeling_gemma.eager_attention_forward(
                self.paligemma.language_model.layers[layer_idx].self_attn,
                query_states,
                key_states,
                value_states,
                layer_attention_mask,
                scaling,
            )
            head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
            att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

            outputs_embeds = []
            start_pos = 0
            for i, hidden_states in enumerate(layer_inputs):
                layer = models[i].layers[layer_idx]
                end_pos = start_pos + hidden_states.shape[1]
                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
                out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])
                after_first_residual = out_emb.clone()
                out_emb, gate = layer.post_attention_layernorm(
                    out_emb, cond=layer_adarms_cond[i]
                )
                if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                    out_emb = out_emb.to(dtype=torch.bfloat16)
                out_emb = layer.mlp(out_emb)
                out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)
                outputs_embeds.append(out_emb)
                start_pos = end_pos
            return outputs_embeds

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing and stride > 0 and layer_idx % stride == 0:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_layer_complete,
                    layer_idx,
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                inputs_embeds = compute_layer_complete(
                    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                )

        def compute_final_norms(layer_inputs, layer_adarms_cond):
            outputs_embeds = []
            for i, hidden_states in enumerate(layer_inputs):
                out_emb, _ = models[i].norm(hidden_states, cond=layer_adarms_cond[i])
                outputs_embeds.append(out_emb)
            return outputs_embeds

        if use_gradient_checkpointing and stride > 0:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms,
                inputs_embeds,
                adarms_cond,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

        return [outputs_embeds[0], outputs_embeds[1]], None

    return wrapped
