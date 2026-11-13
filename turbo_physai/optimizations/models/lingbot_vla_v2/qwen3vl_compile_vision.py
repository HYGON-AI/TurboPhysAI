# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Qwen3-VL vision 塔逐 block 编译"""

from __future__ import annotations

import functools
from typing import Optional

import torch

COMPILE_OPTIONS = {"dynamic": True}


def _drop_view(tensor: torch.Tensor) -> torch.Tensor:
    if getattr(tensor, "_base", None) is not None:
        tensor = tensor.clone()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _compile_block_forward(blk):
    """编译一个 block 的 forward 并缓存在实例上。"""
    if torch.compiler.is_compiling():
        return blk.forward
    cached = blk.__dict__.get("_tp_compiled_forward")
    if cached is None:
        cached = torch.compile(blk.forward, **COMPILE_OPTIONS)
        blk.__dict__["_tp_compiled_forward"] = cached
        blk.__dict__["forward"] = cached
    return cached


def compile_vision_blocks(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor = None,
    pos_embeds: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)

    if position_embeddings is None or cu_seqlens is None or max_seqlen is None:
        _, position_embeddings, cu_seqlens, _, max_seqlen = self.preprcess_grid_thw(grid_thw)
    if pos_embeds is None:
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)

    hidden_states = hidden_states + pos_embeds
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        _compile_block_forward(blk)
        hidden_states = blk(
            _drop_view(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
            **kwargs,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[
                self.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


def install_on_block_init(original, _options=None):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for blk in self.blocks:
            _compile_block_forward(blk)
        return None

    return wrapped


class DisabledStepCounter:
    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    @torch._dynamo.disable
    def __iadd__(self, other):
        self.value += int(other)
        return self

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return f"DisabledStepCounter({self.value})"


def wrap_init(original, _options=None):
    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.__dict__["steps"] = DisabledStepCounter(self.__dict__["steps"])
        return None

    return wrapped
