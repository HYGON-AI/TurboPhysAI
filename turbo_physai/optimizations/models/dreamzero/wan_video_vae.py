"""Use channels-last memory format in the Wan video VAE.

"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

CACHE_T = 2


def resample_forward(self, x, feat_cache=None, feat_idx=[0]):
    b, c, t, h, w = x.size()
    if self.mode == 'upsample3d':
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = 'Rep'
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[
                        idx] is not None and feat_cache[idx] != 'Rep':
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2),
                        cache_x,
                    ], dim=2)
                if cache_x.shape[2] < 2 and feat_cache[
                        idx] is not None and feat_cache[idx] == 'Rep':
                    cache_x = torch.cat([
                        torch.zeros_like(cache_x),
                        cache_x,
                    ], dim=2)
                if feat_cache[idx] == 'Rep':
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                3)
                x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    x = x.contiguous(memory_format=torch.channels_last)
    x = self.resample(x)
    x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

    if self.mode == 'downsample3d':
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(
                    torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x


def attention_block_forward(self, x):
    identity = x
    b, c, t, h, w = x.size()
    x = rearrange(x, 'b c t h w -> (b t) c h w')
    x = x.contiguous(memory_format=torch.channels_last)
    x = self.norm(x)
    # compute query, key, value
    q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(
        0, 1, 3, 2).contiguous().chunk(3, dim=-1)

    # apply attention
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        #attn_mask=block_causal_mask(q, block_size=h * w)
    )
    x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

    # output
    x = self.proj(x)
    x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
    return x + identity


def encode(self, videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
    hidden_states = []
    for video in videos:
        video = video.unsqueeze(0).to(memory_format=torch.channels_last_3d)
        if tiled:
            tile_size = (tile_size[0] * self.upsampling_factor, tile_size[1] * self.upsampling_factor)
            tile_stride = (tile_stride[0] * self.upsampling_factor, tile_stride[1] * self.upsampling_factor)
            hidden_state = self.tiled_encode(video, tile_size, tile_stride)
        else:
            hidden_state = self.single_encode(video)
        hidden_state = hidden_state.squeeze(0)
        hidden_states.append(hidden_state)
    hidden_states = torch.stack(hidden_states)
    return hidden_states


def init_channels_last(original, options):
    del options

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        for module in self.model.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                module.to(memory_format=torch.channels_last_3d)
            elif isinstance(module, nn.Conv2d):
                module.to(memory_format=torch.channels_last)

    return __init__


# Compile the encoder used by each VAE encode chunk.
_COMPILE_DEFAULTS = {"dynamic": True}


def _compile_block_forward(original, options):
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    compile_kwargs = {**_COMPILE_DEFAULTS, **(options or {})}
    compiled = torch.compile(original, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped


def compile_encoder3d_forward(original, options):
    return _compile_block_forward(original, options)


def compile_encoder3d_38_forward(original, options):
    return _compile_block_forward(original, options)
