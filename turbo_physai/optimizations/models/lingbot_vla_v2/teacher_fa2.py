# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Depth teacher：DINOv2 attention 走 flash-attn 2"""

from __future__ import annotations

import functools
import types

import torch


def _fa2_forward(self, x: torch.Tensor, attn_bias=None) -> torch.Tensor:
    """Attention.forward 的 SDPA 换成 flash_attn_func"""
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)  # (B, H, N, D)

    if (
        attn_bias is not None
        or not q.is_cuda
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.shape[-1] % 8 != 0
    ):
        raise RuntimeError(
            "lingbot-vla-v2.teacher-fa2: flash-attn 2 cannot handle this attention input"
        )

    from flash_attn import flash_attn_func

    out = flash_attn_func(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        causal=False,
        softmax_scale=self.scale,
    )
    out = out.reshape(B, N, C)

    out = self.proj(out)
    out = self.proj_drop(out)
    return out


def _install_fa2_attention(model) -> int:
    installed = 0
    for module in model.modules():
        if getattr(module, "_tp_fa2_installed", False):
            continue
        if not all(
            hasattr(module, name)
            for name in ("qkv", "proj", "proj_drop", "num_heads", "scale")
        ):
            continue
        module.forward = types.MethodType(_fa2_forward, module)
        module._tp_fa2_installed = True
        installed += 1
    return installed


def wrap_build_depth_model(original, _options=None):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        models = result if isinstance(result, tuple) else (result,)
        for model in models:
            if _install_fa2_attention(model) == 0:
                raise RuntimeError(
                    "lingbot-vla-v2.teacher-fa2: no DINOv2 attention module found"
                )
        return result

    return wrapped
