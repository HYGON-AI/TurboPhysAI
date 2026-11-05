# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from typing import TYPE_CHECKING, Tuple, Optional

from openwam.model.video_backbone.wan.preprocess import check_resize_height_width
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan_backbone import _truncate_padded_context

from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.attention import fa2_padding
from turbo_physai.optimizations.models.openwam.sac import (
    active_ffn_boundary,
    run_attention,
    run_ffn,
)

logger = logging.getLogger(__name__)


def masked_mha(q: Tensor, k: Tensor, v: Tensor, num_heads: int, mask: Optional[Tensor]):
    """(B, Sq, H*D) x (B, Sk, H*D) masked SDPA -> (B, Sq, H*D), using fa2_padding when available."""
    if enabled("fa2_padding"):
        padded = fa2_padding(q, k, v, mask, num_heads)
        if padded is not None:
            return padded
    from openwam.model.kernels.attention import sdpa

    q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
    if mask is not None and mask.dim() == 3:
        mask = mask.unsqueeze(1)
    x = sdpa(q, k, v, attn_mask=mask)
    return rearrange(x, "b n s d -> b s (n d)", n=num_heads)


def wan_post(block, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp, attn_out, context, context_mask, attention):
    sa, ca = block.self_attn, block.cross_attn
    from openwam.model.kernels.fused_blocks import _compiled, _wan_post_a
    hs, cq = _compiled("wan_post_a", _wan_post_a)(
        residual_x,
        gate_msa,
        attn_out,
        sa.o.weight,
        sa.o.bias,
        block.norm3.weight,
        block.norm3.bias,
        ca.q.weight,
        ca.q.bias,
        ca.norm_q.weight,
        float(block.norm3.eps),
        float(ca.norm_q.eps),
    )
    ck = ca.norm_k(ca.k(context))
    cv = ca.v(context)
    cx = run_attention(
        attention,
        cq,
        ck,
        cv,
        ca.num_heads,
        context_mask,
        label="video_cross_attention",
    )
    if active_ffn_boundary():
        hs = hs + F.linear(cx, ca.o.weight, ca.o.bias)
        m = F.layer_norm(hs, (hs.shape[-1],), None, None, float(block.norm2.eps)) * (1 + scale_mlp) + shift_mlp
        ffn_out = run_ffn(block.ffn, m, video=True)
        return hs + gate_mlp * ffn_out
    from openwam.model.kernels.fused_blocks import _post_b
    f1, f2 = block.ffn[0], block.ffn[2]
    return _compiled("post_b", _post_b)(
        hs,
        cx,
        ca.o.weight,
        ca.o.bias,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        f1.weight,
        f1.bias,
        f2.weight,
        f2.bias,
        float(block.norm2.eps),
        None,
        None,
    )


def WanBase_preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
    device = self.device
    dtype = self.dtype

    video_latents = kw.get("video_latents")
    if video_latents is not None:
        num_frames, height, width = kw["video_size"]
    else:
        uint8_clips = isinstance(frames[0], Tensor)
        if uint8_clips:
            clip_t, clip_h, clip_w = int(frames[0].shape[0]), int(frames[0].shape[1]), int(frames[0].shape[2])
        else:
            clip_t, clip_h, clip_w = len(frames[0]), frames[0][0].size[1], frames[0][0].size[0]
        height, width, num_frames = check_resize_height_width(
            clip_h,
            clip_w,
            clip_t,
            height_division_factor=self._height_division_factor,
            width_division_factor=self._width_division_factor,
            time_division_factor=self._time_division_factor,
            time_division_remainder=self._time_division_remainder,
        )

    batch_size = len(frames)
    has_text_store = getattr(self.text_encoder, "_openwam_text_store", None) is not None
    text_cache = self.__dict__.get("_train_text_cache")
    if (
        not has_text_store
        and text_cache is None
        and os.environ.get("OPENWAM_TEXT_CACHE", "0") == "1"
    ):
        text_cache = wan_encode.TrainTextEmbedCache(
            max_entries=int(os.environ.get("OPENWAM_TEXT_CACHE_MAX_ENTRIES", "256"))
        )
        self.__dict__["_train_text_cache"] = text_cache
    if (
        not has_text_store
        and text_cache is not None
        and self._text_encoder_offload
        and not hasattr(text_cache, "offload")
    ):
        text_cache.offload = wan_encode.TextEncoderOffload(self.text_encoder)

    def _encode_text():
        if text_cache is None:
            ctx, sl = wan_encode.encode_text(
                text,
                tokenizer=self._tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
            )
            if has_text_store and self._variant.supports_context_truncation:
                ctx = _truncate_padded_context(
                    ctx, int(sl.max()) if len(sl) else 0
                )
            return ctx, sl
        text_cache.last_max_len = None
        ctx, sl = wan_encode.encode_text(
            text,
            tokenizer=self._tokenizer,
            text_encoder=self.text_encoder,
            device=self.device,
            cache=text_cache,
        )
        if self._variant.supports_context_truncation:
            ctx = _truncate_padded_context(ctx, text_cache.last_max_len)
        return ctx, sl

    video_first = video_latents is None and (
        getattr(self, "_openwam_video_first", False) or os.environ.get("OPENWAM_VAE_FIRST", "0") == "1"
    )
    if not video_first:
        context, seq_lens = _encode_text()

    stacked_inputs = None
    if video_latents is not None:
        input_latents = video_latents.to(dtype=dtype, device=device)
    else:
        if uint8_clips and self.video_encoder is None:
            stacked_inputs = wan_encode.preprocess_video(
                list(frames), encoder=None, dtype=self.dtype, device=self.device
            )
        else:
            all_input_videos = []
            for clip_frames in frames:
                all_input_videos.append(
                    wan_encode.preprocess_video(
                        clip_frames, encoder=self.video_encoder, dtype=self.dtype, device=self.device
                    )
                )
            stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = wan_encode.encode_video(
            stacked_inputs, vae=self.vae, encoder=self.video_encoder
        )
        input_latents = input_latents.to(dtype=dtype, device=device)

    if video_first:
        context, seq_lens = _encode_text()

    cond = self._variant.build_train_conditioning(
        self,
        input_latents=input_latents,
        frames=frames,
        ref_images=kw.get("ref_images"),
        vace_videos=kw.get("vace_videos"),
        stacked_inputs=stacked_inputs,
        B=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        first_frame_image=kw.get("first_frame_image"),
    )

    return {
        "input_latents": input_latents,
        "context": context,
        "seq_lens": seq_lens,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "vace_context": cond.get("vace_context"),
        "fuse_vae_embedding_in_latents": cond.get("fuse_vae_embedding_in_latents", False),
        "num_clean_prefix_frames": cond.get("num_clean_prefix_frames", 0),
        "first_frame_latents": cond.get("first_frame_latents"),
        "clip_feature": cond.get("clip_feature"),
        "y": cond.get("y"),
    }


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

masked_mha_wrapper = _port_wrapper(
    masked_mha,
    "openwam.model.kernels.fused_blocks.masked_mha",
    "openwam.compute",
)

wan_post_wrapper = _port_wrapper(
    wan_post,
    "openwam.model.kernels.fused_blocks.wan_post",
    "openwam.compute",
)

WanBase_preprocess_input_for_train_wrapper = _port_wrapper(
    WanBase_preprocess_input_for_train,
    "openwam.model.video_backbone.wan_backbone.WanBase.preprocess_input_for_train",
    "openwam.compute",
)
