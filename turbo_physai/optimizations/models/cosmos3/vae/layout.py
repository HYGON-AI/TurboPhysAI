# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Wan VAE channels-last layout + fused causal Conv3d (adapted source port).

Replaces the tokenizer's layout helpers and the layer forwards that feed them.
``CausalConv3d._fused_path_enabled`` is installed as a NEW method on the class
(the upstream class has no such attribute), and ``_update_cache_and_apply`` /
``CausalConv3d.forward`` route through the fused
``fused_causal_cache_pad_conv3d_ndhwc`` op when eligible.  Defaults follow the
source: the fused path stays off unless ``COSMOS_WAN_VAE_CLEAN_FUSED_OP=1`` and
channels-last activates automatically on HIP only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 as _vae_mod

from .._source_binding import bind_missing_globals
from .fused_ops import (
    conv_channels_supported,
    fused_causal_cache_pad_conv3d_ndhwc,
    fused_dtype_supported,
    fused_ops_enabled,
)

try:
    from cosmos_framework.utils.memory_format import training_channels_last_enabled
except ModuleNotFoundError as exc:
    if exc.name != "cosmos_framework.utils.memory_format":
        raise
    from ....common.pytorch.memory_format import training_channels_last_enabled

CACHE_T = 2

# HIP uses channels-last; non-HIP retains the historical layout.
_USE_CHANNELS_LAST = training_channels_last_enabled()


def _to_channels_last_2d(t: torch.Tensor) -> torch.Tensor:
    if _USE_CHANNELS_LAST and t.dim() == 4:
        return t.contiguous(memory_format=torch.channels_last)
    return t


def _to_channels_last_3d(t: torch.Tensor) -> torch.Tensor:
    if _USE_CHANNELS_LAST and t.dim() == 5:
        return t.contiguous(memory_format=torch.channels_last_3d)
    return t


def _is_layout_contiguous(t: torch.Tensor) -> bool:
    if _USE_CHANNELS_LAST and t.dim() == 5:
        return t.is_contiguous(memory_format=torch.channels_last_3d)
    return t.is_contiguous()


def _contiguous_clone(t: torch.Tensor) -> torch.Tensor:
    """Return a contiguous copy of *t* using exactly one allocation."""
    if _USE_CHANNELS_LAST and t.dim() == 5:
        return t.clone(memory_format=torch.channels_last_3d)
    if t.is_contiguous():
        return t.clone()
    return t.contiguous()


def _update_cache_and_apply(
    x: torch.Tensor,
    layer: "CausalConv3d",
    feat_cache: list,
    feat_idx: list[int],
) -> torch.Tensor:
    """Apply a CausalConv3d with temporal cache management.

    Saves the last CACHE_T frames of ``x`` as the new cache entry and,
    when the current chunk has fewer than 2 frames, prepends the last
    cached frame so the cache always spans 2 frames.

    Note that feat_idx is a list with a single element, which stores
    the index of the current CausalConv3d layer. List is used here so
    feat_idx can be mutated in place, and the caller can pass in a reference
    to the list.
    """
    idx = feat_idx[0]
    previous_cache = feat_cache[idx]
    cache_for_op = previous_cache
    if cache_for_op is not None:
        cache_for_op = cache_for_op.to(x.device)
        if layer._padding[4] <= 0:
            cache_for_op = None

    if causal_conv3d_fused_path_enabled(layer, x, cache_for_op):
        pre_padding = (layer._padding[4], layer._padding[2], layer._padding[0])
        post_padding = (layer._padding[5], layer._padding[3], layer._padding[1])
        x, cache_x = fused_causal_cache_pad_conv3d_ndhwc(
            x,
            cache_for_op,
            layer.weight,
            layer.bias,
            pre_padding=pre_padding,
            post_padding=post_padding,
            stride=layer.stride,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        if cache_x.shape[2] < 2 and previous_cache is not None:
            cache_x = torch.cat(
                [previous_cache[:, :, -1:, :, :].to(cache_x.device), cache_x],
                dim=2,
            ).contiguous(memory_format=torch.channels_last_3d)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
        return x

    cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])
    if cache_x.shape[2] < 2 and previous_cache is not None:
        cache_x = torch.cat(
            [
                previous_cache[:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                cache_x,
            ],
            dim=2,
        )  # [B,C,2,H,W]
    x = layer(x, previous_cache)
    feat_cache[idx] = cache_x
    feat_idx[0] += 1
    return x


def causal_conv3d_fused_path_enabled(self, x: torch.Tensor, cache_x: torch.Tensor | None) -> bool:
    return (
        fused_ops_enabled()
        and _USE_CHANNELS_LAST
        and fused_dtype_supported(x.dtype)
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and self.weight.is_contiguous(memory_format=torch.channels_last_3d)
        and (cache_x is None or cache_x.is_contiguous(memory_format=torch.channels_last_3d))
        and (cache_x is None or cache_x.dtype == x.dtype)
        and self.kernel_size == (3, 3, 3)
        and conv_channels_supported(self.in_channels, self.out_channels)
        and self.stride == (1, 1, 1)
        and self.dilation == (1, 1, 1)
        and self._padding == (1, 1, 1, 1, 2, 0)
        and self.groups == 1
    )


def causal_conv3d_forward(self, x, cache_x=None):  # x: [B,C,T,H,W]
    if cache_x is not None:
        cache_x = cache_x.to(x.device)
        if self._padding[4] <= 0:
            cache_x = None

    if causal_conv3d_fused_path_enabled(self, x, cache_x):
        pre_padding = (self._padding[4], self._padding[2], self._padding[0])
        post_padding = (self._padding[5], self._padding[3], self._padding[1])
        out, _ = fused_causal_cache_pad_conv3d_ndhwc(
            x,
            cache_x,
            self.weight,
            self.bias,
            pre_padding=pre_padding,
            post_padding=post_padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )
        return out

    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)  # [B,C,T+cache_T,H,W]
        padding[4] -= cache_x.shape[2]
    x = F.pad(x, padding)  # [B,C,T_padded,H_padded,W_padded]
    x = _to_channels_last_3d(x)

    # Zero-arg super() cannot be used outside the class body; the target class
    # subclasses nn.Conv3d directly, so call its forward explicitly.
    return nn.Conv3d.forward(self, x)  # [B,out_C,T_out,H_out,W_out]


# The extracted methods call self._fused_path_enabled(...); the catalog installs
# causal_conv3d_fused_path_enabled under that attribute name on CausalConv3d.

def resample_forward(self, x, feat_cache=None, feat_idx=[0]):  # x: [B,C,T,H,W]
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # First frame: skip time_conv, seed cache with zeros so the next call sees a real tensor
                feat_cache[idx] = torch.zeros(b, c, CACHE_T, h, w, device=x.device, dtype=x.dtype)  # [B,C,2,H,W]
                feat_idx[0] += 1
            else:
                cache_x = _contiguous_clone(x[:, :, -CACHE_T:, :, :])  # [B,C,<=2,H,W]
                if cache_x.shape[2] < 2:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                            cache_x,
                        ],
                        dim=2,
                    )  # [B,C,2,H,W]
                x = self.time_conv(x, feat_cache[idx])  # [B,C*2,T,H,W]
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
                x = x.reshape(b, 2, c, t, h, w)  # [B,2,C,T,H,W]
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)  # [B,C,T,2,H,W]
                x = x.reshape(b, c, t * 2, h, w)  # [B,C,T*2,H,W]
    t = x.shape[2]
    x = _to_channels_last_2d(rearrange(x, "b c t h w -> (b t) c h w"))  # [B*T,C,H,W]
    x = self.resample(x)  # [B*T,C_out,H_out,W_out]
    x = rearrange(x, "(b t) c h w -> b c t h w", t=t)  # [B,C_out,T,H_out,W_out]

    if self.mode == "downsample3d":
        # Important for torch.compile: when we're *not* doing streaming/cache-based inference
        # (feat_cache is None), we still need to apply the temporal downsample conv.
        if feat_cache is None:
            # `time_conv` has kernel (3,1,1), stride 2 in time, and no internal temporal padding.
            # In the streaming path, we effectively provide left temporal context via cached frames.
            # For the non-streaming path, pad 2 frames on the left so:
            # - the conv is always valid (T>=3)
            # - the output temporal length matches the shortcut path's ceil(T/2) behavior
            x = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H_out,W_out]
            x = self.time_conv(x)  # [B,C,T//2+1,H_out,W_out]
        else:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                # First call for this layer in a streaming/windowed pass.
                # The baseline streaming path primes caches with a single-frame chunk (T==1),
                # where skipping time_conv preserves both correctness and shape alignment.
                #
                # If this is ever called with T>1 (non-standard chunking), fall back to a padded
                # time_conv so the main path stays compatible with the shortcut path.
                if x.shape[2] == 1:
                    feat_cache[idx] = _contiguous_clone(x)
                else:
                    cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H_out,W_out]
                    x_in = F.pad(x, (0, 0, 0, 0, 2, 0))  # [B,C,T+2,H_out,W_out]
                    x = self.time_conv(x_in)  # [B,C,T//2+1,H_out,W_out]
                    feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                cache_x = _contiguous_clone(x[:, :, -1:, :, :])  # [B,C,1,H_out,W_out]
                x_cat = torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)  # [B,C,T+1,H_out,W_out]
                t_cat = x_cat.shape[2]
                if t_cat < 3:
                    x_cat = F.pad(x_cat, (0, 0, 0, 0, 3 - t_cat, 0))  # [B,C,3,H_out,W_out]
                x = self.time_conv(x_cat)  # [B,C,T//2+1,H_out,W_out]
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x


def attention_block_forward(self, x):  # x: [B,C,T,H,W]
    identity = x
    b, c, t, h, w = x.size()
    x = _to_channels_last_2d(rearrange(x, "b c t h w -> (b t) c h w"))  # [B*T,C,H,W]
    x = self.norm(x)  # [B*T,C,H,W]
    # compute query, key, value
    q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
    # q,k,v: [B*T,1,H*W,C]

    # apply attention
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
    )  # [B*T,1,H*W,C]
    x = x.squeeze(1).permute(0, 2, 1).contiguous().reshape(b * t, c, h, w)  # [B*T,C,H,W]

    # output
    x = self.proj(_to_channels_last_2d(x))  # [B*T,C,H,W]
    x = rearrange(x, "(b t) c h w-> b c t h w", t=t)  # [B,C,T,H,W]
    return x + identity  # [B,C,T,H,W]


def avg_down3d_forward(
    self, x: torch.Tensor
) -> torch.Tensor:  # x: [B,C,T,H,W] -> [B,out_channels,T//factor_t,H//factor_s,W//factor_s]
    pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
    pad = (0, 0, 0, 0, pad_t, 0)
    x = F.pad(x, pad).contiguous()  # [B,C,T_padded,H,W]
    B, C, T, H, W = x.shape
    x = x.view(
        B,
        C,
        T // self.factor_t,
        self.factor_t,
        H // self.factor_s,
        self.factor_s,
        W // self.factor_s,
        self.factor_s,
    )  # [B,C,T//ft,ft,H//fs,fs,W//fs,fs]
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()  # [B,C,ft,fs,fs,T//ft,H//fs,W//fs]
    x = x.view(
        B,
        C * self.factor,
        T // self.factor_t,
        H // self.factor_s,
        W // self.factor_s,
    )  # [B,C*factor,T//ft,H//fs,W//fs]
    x = x.view(
        B,
        self.out_channels,
        self.group_size,
        T // self.factor_t,
        H // self.factor_s,
        W // self.factor_s,
    )  # [B,out_channels,group_size,T//ft,H//fs,W//fs]
    x = x.mean(dim=2)  # [B,out_channels,T//ft,H//fs,W//fs]
    return x


def dup_up3d_forward(
    self, x: torch.Tensor, first_chunk=False
) -> torch.Tensor:  # x: [B,in_channels,T,H,W] -> [B,out_channels,T*factor_t,H*factor_s,W*factor_s]
    x = x.contiguous()
    x = x.repeat_interleave(self.repeats, dim=1)  # [B,in_channels*repeats,T,H,W]
    x = x.view(
        x.size(0),
        self.out_channels,
        self.factor_t,
        self.factor_s,
        self.factor_s,
        x.size(2),
        x.size(3),
        x.size(4),
    )  # [B,out_channels,ft,fs,fs,T,H,W]
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()  # [B,out_channels,T,ft,H,fs,W,fs]
    x = x.view(
        x.size(0),
        self.out_channels,
        x.size(2) * self.factor_t,
        x.size(4) * self.factor_s,
        x.size(6) * self.factor_s,
    )  # [B,out_channels,T*ft,H*fs,W*fs]
    if first_chunk:
        x = x[:, :, self.factor_t - 1 :, :, :]  # [B,out_channels,T*ft-ft+1,H*fs,W*fs]
    return x


bind_missing_globals(globals(), _vae_mod)
