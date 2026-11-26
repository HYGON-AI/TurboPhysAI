# Copyright (c) 2026 The FastWAM Authors
# SPDX-License-Identifier: MIT
# Adapted for TurboPhysAI: package imports, lazy compilation and legacy API compatibility.
# See third_party/licenses/FastWAM-MIT.txt and THIRD_PARTY_NOTICES.md.

import torch
from fastwam.models.wan22.wan_video_vae import patchify
from .compile import compile_hotpath


def encode_videos(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
    if tiled:
        raise NotImplementedError("Tiled encoding is not allowed yet.")
    hidden_states = []
    for video in videos:
        video = video.unsqueeze(0)
        hidden_state = self.single_encode(video, device)
        hidden_state = hidden_state.squeeze(0)
        hidden_states.append(hidden_state)
    hidden_states = torch.stack(hidden_states)
    return hidden_states


@compile_hotpath
def encode_vae38(self, x, scale):
    self.clear_cache()
    x = patchify(x, patch_size=2)
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, :1, :, :],
                               feat_cache=self._enc_feat_map,
                               feat_idx=self._enc_conv_idx)
        else:
            out_, self._enc_feat_map, self._enc_conv_idx = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                                feat_cache=self._enc_feat_map,
                                feat_idx=self._enc_conv_idx)
            out = torch.cat([out, out_], 2)
    mu, log_var = self.conv1(out).chunk(2, dim=1)
    if isinstance(scale[0], torch.Tensor):
        mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
            1, self.z_dim, 1, 1, 1)
    else:
        scale = scale.to(dtype=mu.dtype, device=mu.device)
        mu = (mu - scale[0]) * scale[1]
    self.clear_cache()
    return mu
