# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack

logger = logging.getLogger(__name__)

_QKV_NAMES = ("q_proj", "k_proj", "v_proj")
_FUSED = ".qkv_proj."


class QKVProj(nn.Module):
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        self.out_splits = (q_proj.out_features, k_proj.out_features, v_proj.out_features)
        # 新建 Parameter 的 requires_grad 默认是 True，会静默解冻上游已冻结的 q/k/v
        requires_grad = any(
            p.requires_grad for p in (q_proj.weight, k_proj.weight, v_proj.weight)
        )
        with torch.no_grad():
            weight = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)
            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0)
        self.weight = nn.Parameter(weight, requires_grad=requires_grad)
        self.bias = nn.Parameter(bias, requires_grad=requires_grad)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, self.bias)


def merge_qkv(module: nn.Module) -> None:
    count = 0
    for attn in module.modules():
        if hasattr(attn, "qkv_proj") or not all(
            hasattr(attn, name) for name in ("q_proj", "k_proj", "v_proj")
        ):
            continue
        attn.qkv_proj = QKVProj(attn.q_proj, attn.k_proj, attn.v_proj)
        del attn.q_proj, attn.k_proj, attn.v_proj
        count += 1
    logger.info("lingbot-vla: fused qkv on %d attention modules", count)


def _unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def _split_fused(value: torch.Tensor, splits: Tuple[int, int, int]):
    if isinstance(value, DTensor):
        value = value.redistribute(placements=[Replicate()])
    q_end = splits[0]
    k_end = q_end + splits[1]
    return value[:q_end], value[q_end:k_end], value[k_end:]


def split_qkv_state_dict(state_dict, model: nn.Module):
    """qkv_proj.{weight,bias} -> 官方 q_proj/k_proj/v_proj.{weight,bias}"""
    model = _unwrap(model)
    out = {}
    for key, value in state_dict.items():
        if _FUSED not in key:
            out[key] = value
            continue
        base, name = key.rsplit(_FUSED, 1)
        module_path = f"{base}.qkv_proj"
        parts = _split_fused(value, model.get_submodule(module_path).out_splits)
        for proj, part in zip(_QKV_NAMES, parts):
            out[f"{base}.{proj}.{name}"] = part
    return out


def _match_proj_key(key: str):
    for proj in _QKV_NAMES:
        for name in ("weight", "bias"):
            suffix = f".{proj}.{name}"
            if key.endswith(suffix):
                return key[: -len(suffix)], proj, name
    return None


def merge_qkv_state_dict(state_dict, model: nn.Module):
    """官方 q_proj/k_proj/v_proj.{weight,bias} -> qkv_proj.{weight,bias}"""
    model = _unwrap(model)
    out = {}
    pending = {}
    for key, value in state_dict.items():
        hit = _match_proj_key(key)
        if hit is None or not hasattr(model.get_submodule(hit[0]), "qkv_proj"):
            out[key] = value
            continue
        base, proj, name = hit
        pending.setdefault((base, name), {})[proj] = value
    for (base, name), parts in pending.items():
        ref = getattr(model.get_submodule(f"{base}.qkv_proj"), name)
        full = torch.cat([_full(parts[proj]) for proj in _QKV_NAMES], dim=0)
        out[f"{base}.qkv_proj.{name}"] = _like(full, ref)
    return out


def _full(value):
    return value.full_tensor() if isinstance(value, DTensor) else value


def _like(value: torch.Tensor, ref) -> torch.Tensor:
    """把全量张量切成与 ref 相同的分片布局。
    FSDP2 的分片不均匀（3420 行按 8 卡是 428*7+424），所以这里不能靠
    cat/slice 拼 local 分片，只能先拿全量再按 ref 的 mesh 重新分发。
    """
    if isinstance(ref, DTensor):
        return distribute_tensor(value, ref.device_mesh, ref.placements)
    return value


def _split_full(value: torch.Tensor, splits: Tuple[int, int, int]):
    q_end = splits[0]
    k_end = q_end + splits[1]
    return value[:q_end], value[q_end:k_end], value[k_end:]


def split_qkv_optim_state_dict(osd, model: nn.Module):
    """融合 qkv_proj 的优化器状态 -> 官方 q/k/v 三份，供官方代码加载。"""
    model = _unwrap(model)
    state = {}
    for fqn, entry in osd["state"].items():
        if _FUSED not in fqn:
            state[fqn] = entry
            continue
        base, name = fqn.rsplit(_FUSED, 1)
        module = model.get_submodule(f"{base}.qkv_proj")
        ref = getattr(module, name)
        parts = {proj: {} for proj in _QKV_NAMES}
        for key, value in entry.items():
            if torch.is_tensor(value) and value.dim() > 0:
                for proj, chunk in zip(
                    _QKV_NAMES, _split_full(_full(value), module.out_splits)
                ):
                    parts[proj][key] = _like(chunk, ref)
            else:
                for proj in _QKV_NAMES:
                    parts[proj][key] = value
        for proj in _QKV_NAMES:
            state[f"{base}.{proj}.{name}"] = parts[proj]

    groups = []
    for group in osd["param_groups"]:
        params = []
        for fqn in group["params"]:
            if _FUSED in fqn:
                base, name = fqn.rsplit(_FUSED, 1)
                params.extend(f"{base}.{proj}.{name}" for proj in _QKV_NAMES)
            else:
                params.append(fqn)
        groups.append({**group, "params": params})
    return {**osd, "state": state, "param_groups": groups}


def merge_qkv_optim_state_dict(osd, model: nn.Module):
    """官方 q/k/v 三份优化器状态 -> 融合 qkv_proj，供本实现加载。"""
    model = _unwrap(model)
    merged = {}
    pending = {}
    for fqn, entry in osd["state"].items():
        hit = _match_proj_key(fqn)
        if hit is None or not hasattr(model.get_submodule(hit[0]), "qkv_proj"):
            merged[fqn] = entry
            continue
        base, proj, name = hit
        pending.setdefault((base, name), {})[proj] = entry
    for (base, name), parts in pending.items():
        ref = getattr(model.get_submodule(f"{base}.qkv_proj"), name)
        entry = {}
        for key, value in parts[_QKV_NAMES[0]].items():
            if torch.is_tensor(value) and value.dim() > 0:
                full = torch.cat(
                    [_full(parts[proj][key]) for proj in _QKV_NAMES], dim=0
                )
                entry[key] = _like(full, ref)
            else:
                entry[key] = value
        merged[f"{base}.qkv_proj.{name}"] = entry

    groups = []
    for group in osd["param_groups"]:
        params = []
        for fqn in group["params"]:
            hit = _match_proj_key(fqn)
            if hit is not None and hasattr(model.get_submodule(hit[0]), "qkv_proj"):
                fqn = f"{hit[0]}.qkv_proj.{hit[2]}"
            if fqn not in params:
                params.append(fqn)
        groups.append({**group, "params": params})
    return {**osd, "state": merged, "param_groups": groups}


def wrap_model_state_dict(original, _options=None):
    """DCP 的 ModelState 落盘前转换 key，保证 checkpoint 是官方权重格式。"""
    @functools.wraps(original)
    def wrapped(self):
        state = original(self)
        return {"model": split_qkv_state_dict(state["model"], self.model)}

    return wrapped


def wrap_load_model_state_dict(original, _options=None):
    """加载官方（或本实现保存的）权重时合并回 fused qkv。"""

    @functools.wraps(original)
    def wrapped(self, state_dict):
        merged = dict(state_dict)
        merged["model"] = merge_qkv_state_dict(state_dict["model"], self.model)
        return original(self, merged)

    return wrapped


def wrap_get_optimizer_state_dict(original, _options=None):
    """保存前把 qkv_proj 的优化器状态拆回官方 q/k/v 三份。

    挂在 dcp 的 get/set_optimizer_state_dict 上，只负责 key 转换；state 的
    DTensor 形态由 fused-adamw 的 OptimizerState 包装负责，两者互不引用。
    """

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        return split_qkv_optim_state_dict(original(*args, **kwargs), model)

    return wrapped


def wrap_set_optimizer_state_dict(original, _options=None):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        if "optim_state_dict" in kwargs:
            kwargs["optim_state_dict"] = merge_qkv_optim_state_dict(
                kwargs["optim_state_dict"], model
            )
            return original(*args, **kwargs)
        return original(
            args[0], args[1], merge_qkv_optim_state_dict(args[2], model), *args[3:]
        )

    return wrapped


def wrap_load_model(original, _options=None):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        # init_device="meta"/cpu 时权重是在 build_parallelize_model 里按官方 q/k/v 名字加载的，融合后的 qkv_proj 拿不到权重，会变成未初始化值
        init_device = kwargs.get("init_device", "cuda")
        if init_device != "cuda":
            raise ValueError(
                "lingbot-vla: qkv-fused 只支持 init_device='cuda'"
                f"（当前 {init_device!r}）。meta/cpu 初始化下官方权重按 q/k/v 名字加载，融合后的 qkv_proj 拿不到权重。"
                "请改用 --train.init_device cuda，或关闭 lingbot-vla.qkv-fused。"
            )
        model = original(*args, **kwargs)
        merge_qkv(model)
        return model

    return wrapped


def wrap_build_checkpointer(original, _options=None):
    """qkv-fused 的 key 转换只挂在 dcp 的 ModelState/OptimizerState 上。"""

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        ckpt_manager = kwargs.get("ckpt_manager", args[1] if len(args) > 1 else "dcp")
        if ckpt_manager != "dcp":
            raise ValueError(
                "lingbot-vla: qkv-fused 只支持 ckpt_manager='dcp'"
                f"（当前 {ckpt_manager!r}）。qkv_proj<->q/k/v 的 key 转换只挂在 dcp 的 ModelState/OptimizerState 上，"
                "bytecheckpoint 会写出官方代码无法加载的 checkpoint。"
                "请改用 --train.ckpt_manager dcp，或关闭 lingbot-vla.qkv-fused。"
            )
        return original(*args, **kwargs)

    return wrapped


def layer_forward(
    self,
    hidden_states: torch.Tensor,
    att_output: Optional[torch.Tensor] = None,
    start: Optional[int] = 0,
    end: Optional[int] = 0,
    compute_kqv: bool = False,
    norm_qkv: bool = False,
    output_atten: bool = False,
    ada_cond: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if compute_kqv:
        if ada_cond is not None:
            hidden_states = self.input_layernorm(hidden_states, ada_cond)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        gate = None
        hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

        proj = getattr(self.self_attn, "qkv_proj", None)
        if proj is None:
            # init_device != "cuda" 时没有融合，回退到官方三个独立投影
            q = self.self_attn.q_proj(hidden_states)
            k = self.self_attn.k_proj(hidden_states)
            v = self.self_attn.v_proj(hidden_states)
            qkv = torch.cat([q, k, v], dim=-1)
            out_splits = (q.shape[-1], k.shape[-1], v.shape[-1])
        else:
            qkv = proj(hidden_states)
            out_splits = proj.out_splits
        query_state, key_state, value_state = (
            state.view(hidden_shape)
            for state in qkv.split(out_splits, dim=-1)
        )
        if norm_qkv:
            query_state = self.q_layernorm(query_state)
            key_state = self.k_layernorm(key_state)

        return query_state, key_state, value_state, gate

    elif output_atten:
        if att_output.dtype != self.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
        out_emb = self.self_attn.o_proj(att_output[:, start:end])

        if gate is not None:
            out_emb = out_emb * gate + hidden_states
        else:
            out_emb += hidden_states
        after_first_residual = out_emb.clone()
        if ada_cond is not None:
            out_emb = self.post_attention_layernorm(out_emb, ada_cond)
            after_gate = None
        else:
            out_emb = self.post_attention_layernorm(out_emb)
            after_gate = None
        out_emb = self.mlp(out_emb)

        if after_gate is not None:
            out_emb = out_emb * after_gate + after_first_residual
        else:
            out_emb += after_first_residual

        return out_emb

    else:
        raise ValueError(
            f"Invaild Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2DecoderLayer in LingBot-VLA"
        )


def vl_layer_forward(
    self,
    hidden_states: torch.Tensor,
    att_output: Optional[torch.Tensor] = None,
    start: Optional[int] = 0,
    end: Optional[int] = 0,
    compute_kqv: bool = False,
    norm_qkv: bool = False,
    output_atten: bool = False,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    if compute_kqv:
        hidden_states = self.input_layernorm(hidden_states)
        hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

        proj = getattr(self.self_attn, "qkv_proj", None)
        if proj is None:
            # init_device != "cuda" 时没有融合，回退到官方三个独立投影
            q = self.self_attn.q_proj(hidden_states)
            k = self.self_attn.k_proj(hidden_states)
            v = self.self_attn.v_proj(hidden_states)
            qkv = torch.cat([q, k, v], dim=-1)
            out_splits = (q.shape[-1], k.shape[-1], v.shape[-1])
        else:
            qkv = proj(hidden_states)
            out_splits = proj.out_splits
        query_state, key_state, value_state = (
            state.view(hidden_shape)
            for state in qkv.split(out_splits, dim=-1)
        )
        if norm_qkv:
            query_state = self.q_layernorm(query_state)
            key_state = self.k_layernorm(key_state)

        return query_state, key_state, value_state

    elif output_atten:
        if att_output.dtype != self.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
        out_emb = self.self_attn.o_proj(att_output[:, start:end])

        out_emb += hidden_states
        after_first_residual = out_emb.clone()

        out_emb = self.post_attention_layernorm(out_emb)
        out_emb = self.mlp(out_emb)

        out_emb += after_first_residual

        return out_emb

    else:
        raise ValueError(
            f"Invaild Operation compute_kqv={compute_kqv} and output_atten={output_atten} with Qwen2_5_VLDecoderLayer in LingBot-VLA"
        )
