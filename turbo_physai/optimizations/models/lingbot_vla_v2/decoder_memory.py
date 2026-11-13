# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""联合 decoder 的 forward 重写，只压激活显存，不改其它策略。"""

from __future__ import annotations

import torch


def cat_float(parts):
    """同 dtype 先 cat 再整体 cast，混合 dtype 走官方的逐段转换。"""
    if all(part.dtype == parts[0].dtype for part in parts):
        return torch.cat(parts, dim=1).float()
    return torch.cat([part.float() for part in parts], dim=1)


def local_attention_input(att_output, start, end, layer):
    local = att_output[:, start:end].to(layer.self_attn.q_proj.weight.dtype)
    if local._base is not None:
        local = local.clone(memory_format=torch.contiguous_format)
    elif not local.is_contiguous():
        local = local.contiguous()
    return local


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
    visual_pos_masks=None,
    deepstack_visual_embeds=None,
):
    from lingbotvla.models.vla.lingbot_vla import modeling_lingbot_vla_v2 as baseline

    models = [self.qwenvl.model.language_model, self.qwen_expert.model]
    num_layers = self.qwenvl.config.text_config.num_hidden_layers
    action_num_layers = self.config.qwen_expert_config.num_hidden_layers
    router_logits_list = []
    assert action_num_layers == num_layers, (
        "Action expert and VLM must have the same number of layers "
        f"(got action={action_num_layers}, vlm={num_layers})."
    )

    for layer_idx in range(num_layers):
        queries, keys, values = [], [], []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            if i == 1:
                q, k, v = models[i].layers[layer_idx](
                    hidden_states, compute_kqv=True, ada_cond=ada_cond
                )
            else:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
            queries.append(q)
            keys.append(k)
            values.append(v)
        query_states = cat_float(queries)
        key_states = cat_float(keys)
        value_states = cat_float(values)
        del queries, keys, values, q, k, v
        query_states, key_states = self.apply_mrope(query_states, key_states, position_ids)
        key_states, value_states, past_key_values = self.handle_kv_cache(
            key_states, value_states, layer_idx,
            past_key_values=past_key_values, use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        if self.config.attention_implementation == "flex_cached":
            if layer_idx == 0:
                full_len = query_states.shape[1]
                full_block_mask = baseline.build_block_mask(
                    attention_mask, self.qwenvl.config.text_config.num_attention_heads,
                    full_len, full_len,
                )
            att_output = baseline.flex_attention_with_block_mask(
                query_states, key_states, value_states, full_block_mask, query_states.shape[1]
            )
        else:
            att_output = self.attention_interface(
                query_states, key_states, value_states, attention_mask
            )
        del query_states, key_states, value_states

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            layer = models[i].layers[layer_idx]
            # 必须在 layer 入口前完成，否则 checkpoint 存下的就是视图
            branch_att = local_attention_input(att_output, start, end, layer)
            branch_start, branch_end = 0, branch_att.shape[1]
            if i == 1:
                out_emb, router_logits = layer(
                    hidden_states, branch_att, branch_start, branch_end,
                    output_atten=True, ada_cond=ada_cond,
                )
                if router_logits is not None:
                    router_logits_list.append(router_logits)
            else:
                out_emb = layer(
                    hidden_states, branch_att, branch_start, branch_end, output_atten=True
                )
                out_emb = self._apply_deepstack(
                    out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds
                )
            outputs_embeds.append(out_emb)
            del branch_att
            start = end
        del att_output
        inputs_embeds = outputs_embeds

    outputs_embeds = []
    for i, hidden_states in enumerate(inputs_embeds):
        if hidden_states is None:
            outputs_embeds.append(None)
        elif self.config.final_norm_adanorm and i == 1:
            out_emb, _ = models[i].norm(hidden_states, ada_cond)
            outputs_embeds.append(out_emb)
        else:
            outputs_embeds.append(models[i].norm(hidden_states))
    return outputs_embeds, past_key_values, router_logits_list
