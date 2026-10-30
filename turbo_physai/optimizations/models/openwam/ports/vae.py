# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import torch
import openwam.model.video_backbone.wan.models.vae as _source
import torch.nn as nn
from openwam.model.video_backbone.wan.models.vae import (
    CACHE_T,
    CausalConv3d,
    F,
    RMS_norm,
    ResidualBlock,
    check_is_instance,
    patchify,
    rearrange,
)
from turbo_physai.optimizations.models.openwam import enabled, pointwise


def _layout(x):
    if enabled("vae_channels_last") and x.ndim == 5:
        from turbo_physai.optimizations.models.openwam.vae_layout import channels_last

        return channels_last(x)
    if enabled("vae_channels_last") and x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x


def _clone_cache(x):
    if enabled("vae_channels_last") and x.ndim == 5:
        return x.clone(memory_format=torch.channels_last_3d)
    return x.clone()


def CausalConv3d_forward(self, x, cache_x=None):
    padding = list(self._padding)
    if cache_x is not None and self._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    x = _layout(F.pad(x, padding))

    return _layout(super(_source.CausalConv3d, self).forward(x))


def RMS_norm_forward(self, x):
    if enabled("vae_pointwise_compile"):
        return pointwise.vae_normalize(
            x, self.gamma, self.bias, 1 if self.channel_first else -1, self.scale
        )
    return (
        F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma
        + self.bias
    )


def Resample_forward(self, x, feat_cache=None, feat_idx=[0]):
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] == "Rep"
                ):
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = rearrange(x, "b c t h w -> (b t) c h w")
    x = self.resample(_layout(x))
    x = _layout(rearrange(x, "(b t) c h w -> b c t h w", t=t))

    if self.mode == "downsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = _clone_cache(x)
                feat_idx[0] += 1
            else:
                cache_x = _clone_cache(x[:, :, -1:, :, :])
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x, feat_cache, feat_idx


def ResidualBlock_forward(self, x, feat_cache=None, feat_idx=[0]):
    h = self.shortcut(x)
    skip_silu = False
    for layer_id, layer in enumerate(self.residual):
        if skip_silu:
            skip_silu = False
            continue
        if (
            enabled("vae_pointwise_compile")
            and isinstance(layer, RMS_norm)
            and layer_id + 1 < len(self.residual)
            and isinstance(self.residual[layer_id + 1], nn.SiLU)
        ):
            x = pointwise.vae_normalize(
                x,
                layer.gamma,
                layer.bias,
                1 if layer.channel_first else -1,
                layer.scale,
                apply_silu=True,
            )
            if enabled("vae_channels_last"):
                x = _layout(x)
            skip_silu = True
            continue
        if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
            if enabled("vae_channels_last"):
                x = _layout(x)
            idx = feat_idx[0]
            cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    result = x + h
    if enabled("vae_channels_last"):
        result = _layout(result)
    return result, feat_cache, feat_idx


def Encoder3d_forward(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat(
                [
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    ## downsamples
    for layer in self.downsamples:
        if feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## middle
    for layer in self.middle:
        if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## head
    for layer in self.head:
        if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x, feat_cache, feat_idx


def Encoder3d_38_forward(self, x, feat_cache=None, feat_idx=[0]):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat(
                [
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    ## downsamples
    for layer in self.downsamples:
        if feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## middle
    for layer in self.middle:
        if isinstance(layer, ResidualBlock) and feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## head
    for layer in self.head:
        if isinstance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)

    return x, feat_cache, feat_idx


def Decoder3d_forward(self, x, feat_cache=None, feat_idx=[0]):
    ## conv1
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            # cache last frame of last two chunk
            cache_x = torch.cat(
                [
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    ## middle
    for layer in self.middle:
        if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## upsamples
    for layer in self.upsamples:
        if feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## head
    for layer in self.head:
        if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x, feat_cache, feat_idx


def Decoder3d_38_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat(
                [
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                    cache_x,
                ],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    for layer in self.middle:
        if check_is_instance(layer, ResidualBlock) and feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    ## upsamples
    for layer in self.upsamples:
        if feat_cache is not None:
            x, feat_cache, feat_idx = layer(x, feat_cache, feat_idx, first_chunk)
        else:
            x = layer(x)

    ## head
    for layer in self.head:
        if check_is_instance(layer, CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = _clone_cache(x[:, :, -CACHE_T:, :, :])
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x, feat_cache, feat_idx


def WanVideoVAE_batch_encode(self, videos, device):
    """Encode a pre-stacked (B, C, T, H, W) tensor in one VAE forward pass.

    Unlike ``encode()`` which loops over a list and calls ``single_encode``
    per video, this method passes the full batch tensor directly to the
    inner ``VideoVAE_.encode()``.  The inner encoder iterates over the
    **temporal** dimension (4-frame chunks) but preserves the batch dimension
    through Conv3D, so batched input is natively supported.

    Only supports non-tiled encoding.  Intended for training use where
    ``tiled=False`` and the VAE is frozen (``torch.no_grad``).

    Args:
        videos: (B, 3, T, H, W) pixel-space video tensor, value range [0, 1].
        device: Target device for encoding.

    Returns:
        (B, z_dim, T_lat, H_lat, W_lat) latent tensor, z-score normalized.
    """
    videos = _layout(videos.to(device))
    with torch.no_grad():
        return self.model.encode(
            videos, functools.partial(WanVideoVAE__scale_for, self)(videos)
        )


def WanVideoVAE_prepare_scale_cache(self):
    parameter = next(self.model.parameters())
    for index, scale in enumerate(self.scale):
        self.register_buffer(
            f"_openwam_scale_{index}",
            scale.to(device=parameter.device, dtype=parameter.dtype),
            persistent=False,
        )


def WanVideoVAE__scale_for(self, x):
    if enabled("constant_cache"):
        # Also supports direct encoder calls outside the trainer. Initialize
        # before the encoder launches kernels, then reuse nonpersistent buffers.
        if not hasattr(self, "_openwam_scale_0"):
            functools.partial(WanVideoVAE_prepare_scale_cache, self)()
        return [
            getattr(self, f"_openwam_scale_{i}").to(device=x.device, dtype=x.dtype)
            for i in range(2)
        ]
    return self.scale


def WanVideoVAE_single_encode(self, video, device):
    video = _layout(video.to(device))
    x = self.model.encode(video, functools.partial(WanVideoVAE__scale_for, self)(video))
    return x


def WanVideoVAE_single_decode(self, hidden_state, device):
    hidden_state = hidden_state.to(device)
    video = self.model.decode(
        hidden_state, functools.partial(WanVideoVAE__scale_for, self)(hidden_state)
    )
    return video.clamp_(-1, 1)


def VideoVAE38__encode(self, x, scale):
    self.clear_cache()
    x = _layout(patchify(x, patch_size=2))
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
        else:
            out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
            1, self.z_dim, 1, 1, 1
        )
    else:
        scale = scale.to(dtype=mu.dtype, device=mu.device)
        mu = (mu - scale[0]) * scale[1]
    self.clear_cache()
    return mu


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

CausalConv3d_forward_wrapper = _port_wrapper(
    CausalConv3d_forward,
    "openwam.model.video_backbone.wan.models.vae.CausalConv3d.forward",
    "openwam.vae",
)

RMS_norm_forward_wrapper = _port_wrapper(
    RMS_norm_forward,
    "openwam.model.video_backbone.wan.models.vae.RMS_norm.forward",
    "openwam.vae",
)

Resample_forward_wrapper = _port_wrapper(
    Resample_forward,
    "openwam.model.video_backbone.wan.models.vae.Resample.forward",
    "openwam.vae",
)

ResidualBlock_forward_wrapper = _port_wrapper(
    ResidualBlock_forward,
    "openwam.model.video_backbone.wan.models.vae.ResidualBlock.forward",
    "openwam.vae",
)

Encoder3d_forward_wrapper = _port_wrapper(
    Encoder3d_forward,
    "openwam.model.video_backbone.wan.models.vae.Encoder3d.forward",
    "openwam.vae",
)

Encoder3d_38_forward_wrapper = _port_wrapper(
    Encoder3d_38_forward,
    "openwam.model.video_backbone.wan.models.vae.Encoder3d_38.forward",
    "openwam.vae",
)

Decoder3d_forward_wrapper = _port_wrapper(
    Decoder3d_forward,
    "openwam.model.video_backbone.wan.models.vae.Decoder3d.forward",
    "openwam.vae",
)

Decoder3d_38_forward_wrapper = _port_wrapper(
    Decoder3d_38_forward,
    "openwam.model.video_backbone.wan.models.vae.Decoder3d_38.forward",
    "openwam.vae",
)

WanVideoVAE_batch_encode_wrapper = _port_wrapper(
    WanVideoVAE_batch_encode,
    "openwam.model.video_backbone.wan.models.vae.WanVideoVAE.batch_encode",
    "openwam.vae",
)

WanVideoVAE_single_encode_wrapper = _port_wrapper(
    WanVideoVAE_single_encode,
    "openwam.model.video_backbone.wan.models.vae.WanVideoVAE.single_encode",
    "openwam.vae",
)

WanVideoVAE_single_decode_wrapper = _port_wrapper(
    WanVideoVAE_single_decode,
    "openwam.model.video_backbone.wan.models.vae.WanVideoVAE.single_decode",
    "openwam.vae",
)

VideoVAE38__encode_wrapper = _port_wrapper(
    VideoVAE38__encode,
    "openwam.model.video_backbone.wan.models.vae.VideoVAE38_.encode",
    "openwam.vae",
)
