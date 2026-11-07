# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import torch
from einops import rearrange
from torch import Tensor
from typing import Optional

from openwam.model.video_backbone.wan.preprocess import check_resize_height_width
from openwam.model.video_backbone.wan import encode as wan_encode
from .encode import prepare_text_cached

from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.attention import fa2_padding

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


def WanBase_preprocess_input_for_train(self, *, frames=None, text=None, **kw) -> dict:
    """Unified train preprocessing: raw data (``frames``/``text``, optional
    ``vace_videos``/``ref_images`` in kw) → the denoising-loop input dict.
    """
    device = self.device
    dtype = self.dtype

    video_latents = kw.get("video_latents")
    if video_latents is not None:
        num_frames, height, width = kw["video_size"]
    else:
        uint8_clips = isinstance(frames[0], Tensor)
        if uint8_clips:
            # (T, H, W, 3) uint8 clips from openwam.dataloader.collate.openwam_collate
            clip_t, clip_h, clip_w = (
                int(frames[0].shape[0]),
                int(frames[0].shape[1]),
                int(frames[0].shape[2]),
            )
        else:
            clip_t, clip_h, clip_w = (
                len(frames[0]),
                frames[0][0].size[1],
                frames[0][0].size[0],
            )
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
    from turbo_physai.optimizations.models.openwam import text_store

    has_text_store = (
        getattr(self.text_encoder, "_openwam_text_store", None) is not None
        or text_store.requested()
    )
    text_cache = self.__dict__.get("_train_text_cache")
    if (
        text_cache is None
        and os.environ.get("OPENWAM_TEXT_CACHE", "0") == "1"
        and not has_text_store
    ):
        text_cache = wan_encode.TrainTextEmbedCache(
            max_entries=int(os.environ.get("OPENWAM_TEXT_CACHE_MAX_ENTRIES", "256"))
        )
        self.__dict__["_train_text_cache"] = text_cache
    if (
        text_cache is not None
        and self._text_encoder_offload
        and not has_text_store
        and not hasattr(text_cache, "offload")
    ):
        # umT5 weights in pinned host memory between cache misses (see wan_encode.TextEncoderOffload)
        text_cache.offload = wan_encode.TextEncoderOffload(self.text_encoder)
    elif (
        text_cache is None
        and self._text_encoder_offload
        and not has_text_store
        and not self.__dict__.get("_offload_warned")
    ):
        self.__dict__["_offload_warned"] = True
        logger.warning(
            "training.text_encoder_offload=true needs OPENWAM_TEXT_CACHE=1 "
            "(umT5 runs every iteration without it); ignored"
        )

    def _encode_text(prepared=None):
        if text_cache is None:  # OPENWAM_TEXT_CACHE off: the default encode
            ctx, sl = wan_encode.encode_text(
                text,
                tokenizer=self._tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
            )
            if self._variant.supports_context_truncation and sl is not None:
                max_len = (
                    int(sl.max().item()) if isinstance(sl, torch.Tensor) else max(sl)
                )
                ctx = _truncate_padded_context(ctx, max_len)
            return ctx, sl
        if prepared is None:
            text_cache.last_max_len = None  # None = keep the padded context
            ctx, sl = wan_encode.encode_text(
                text,
                tokenizer=self._tokenizer,
                text_encoder=self.text_encoder,
                device=self.device,
                cache=text_cache,
            )
        else:
            ctx, sl = prepared()
        if self._variant.supports_context_truncation:
            ctx = _truncate_padded_context(ctx, text_cache.last_max_len)
        return ctx, sl

    video_first = video_latents is None and (
        os.environ.get("OPENWAM_VAE_FIRST", "0") == "1"
        or getattr(self, "_openwam_video_first", False)
    )
    prepared_text = None
    if video_first and text_cache is not None and not has_text_store:
        text_cache.last_max_len = None
        prepared_text = prepare_text_cached(
            text,
            tokenizer=self._tokenizer,
            text_encoder=self.text_encoder,
            device=self.device,
            cache=text_cache,
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
                        clip_frames,
                        encoder=self.video_encoder,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
            stacked_inputs = torch.cat(all_input_videos, dim=0)
        input_latents = wan_encode.encode_video(
            stacked_inputs, vae=self.vae, encoder=self.video_encoder
        )
        input_latents = input_latents.to(dtype=dtype, device=device)

    if video_first:
        context, seq_lens = _encode_text(prepared_text)

    # Variant-specific first-frame / control conditioning lives in
    # ``self._variant``, so this method carries no per-variant ``if``.
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
        "fuse_vae_embedding_in_latents": cond.get(
            "fuse_vae_embedding_in_latents", False
        ),
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


WanBase_preprocess_input_for_train_wrapper = _port_wrapper(
    WanBase_preprocess_input_for_train,
    "openwam.model.video_backbone.wan_backbone.WanBase.preprocess_input_for_train",
    "openwam.compute",
)


def _truncate_padded_context(context: Tensor, max_len: Optional[int]) -> Tensor:
    """Drop trailing all-padding text positions (``OPENWAM_CONTEXT_TRUNC=1``).

    Callers must ensure that every consumer masks positions ``>= seq_len``
    (``context_mask = positions < seq_lens``). VACE's unmasked hint blocks do
    not satisfy this contract. For masked consumers, the cross-attentions carry
    no positional encoding on keys, so keeping the first ``L`` positions with
    ``L >= max(seq_lens)`` leaves the function unchanged while the per-token text
    embedding, K/V projections and norms stop running on hundreds of padding rows.
    """
    if max_len is None or os.environ.get("OPENWAM_CONTEXT_TRUNC", "0") != "1":
        return context
    keep = max(1, int(max_len))
    if keep >= context.shape[1]:
        return context
    return context[:, :keep].contiguous()
