# Copyright 2025 DiT4DiT team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT
# Adapted from DiT4DiT 3a0c3c5; see THIRD_PARTY_NOTICES.md.

"""Cosmos caches share one mixin; original construction/checkpoint layout stays intact."""

from __future__ import annotations

from typing import List, Union
import torch


class _Caches:
    def __init__(self, *args, **kwargs):
        self._latent_stats_cache = {}
        self._condition_tensor_cache = {}
        self._forward_static_cache = {}
        self._token_ids_cache = {}
        self._prompt_embeds_cache = {}
        self._text_encoder_output_cache_safe = None
        self._direct_hidden_capture = False
        super().__init__(*args, **kwargs)
        for name in ("latents_mean", "latents_std"):
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                delattr(self, name)
                self.register_buffer(name, value.detach(), persistent=False)

    def _apply(self, fn):
            # Cached device/dtype views become stale when the module is moved.
            for cache_name in (
                "_latent_stats_cache",
                "_condition_tensor_cache",
                "_forward_static_cache",
                "_prompt_embeds_cache",
            ):
                cache = getattr(self, cache_name, None)
                if cache is not None:
                    cache.clear()
            return super()._apply(fn)

    def _get_latent_stats(
            self, *, device: torch.device, dtype: torch.dtype
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.latents_mean is None or self.latents_std is None:
                raise ValueError(
                    "VAE configuration must define both `latents_mean` and `latents_std`."
                )
            device = torch.device(device)
            key = (device.type, device.index, dtype)
            cached = self._latent_stats_cache.get(key)
            if cached is None:
                cached = (
                    self.latents_mean.to(device=device, dtype=dtype),
                    self.latents_std.to(device=device, dtype=dtype),
                )
                self._latent_stats_cache[key] = cached
            return cached

    def _get_condition_tensors(
            self,
            *,
            batch_size: int,
            num_frames: int,
            height: int,
            width: int,
            num_condition_frames: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return immutable condition tensors for a fixed training shape."""
            device = torch.device(device)
            key = (
                batch_size,
                num_frames,
                height,
                width,
                num_condition_frames,
                device.type,
                device.index,
                dtype,
            )
            cached = self._condition_tensor_cache.get(key)
            if cached is None:
                ones_padding = torch.ones(
                    (batch_size, 1, num_frames, height, width),
                    device=device,
                    dtype=dtype,
                )
                zeros_padding = torch.zeros_like(ones_padding)
                cond_indicator = torch.zeros(
                    (batch_size, 1, num_frames, 1, 1),
                    device=device,
                    dtype=dtype,
                )
                cond_indicator[:, :, :num_condition_frames] = 1.0
                # Keep the original arithmetic exactly; only cache its result.
                cond_mask = (
                    cond_indicator * ones_padding
                    + (1.0 - cond_indicator) * zeros_padding
                )
                cached = (cond_mask, cond_indicator)
                if len(self._condition_tensor_cache) >= 4:
                    self._condition_tensor_cache.clear()
                self._condition_tensor_cache[key] = cached
            return cached

    def _get_forward_static_tensors(
            self,
            *,
            condition_frames: int,
        cond_mask: torch.Tensor,
            cond_indicator: torch.Tensor,
            transformer_dtype: torch.dtype,
            conditional_frame_timestep: float,
            height: int,
            width: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Cache masks and constants that are invariant across micro-steps."""
            device = cond_indicator.device
            key = (
                condition_frames,
            tuple(cond_mask.shape),
                tuple(cond_indicator.shape),
                str(device),
                transformer_dtype,
                float(conditional_frame_timestep),
                height,
                width,
            )
            cached = self._forward_static_cache.get(key)
            if cached is None:
                cond_timestep = (
                    torch.ones_like(cond_indicator)
                    * float(conditional_frame_timestep)
                )
                cond_mask_t = cond_mask.to(transformer_dtype)
                inv_cond_mask_t = 1 - cond_mask_t
                inv_cond_indicator = 1 - cond_indicator
                padding_mask = torch.zeros(
                    (1, 1, height, width),
                    device=device,
                    dtype=transformer_dtype,
                )
                cached = (
                    cond_timestep,
                    cond_mask_t,
                    inv_cond_mask_t,
                    inv_cond_indicator,
                    padding_mask,
                )
                if len(self._forward_static_cache) >= 4:
                    self._forward_static_cache.clear()
                self._forward_static_cache[key] = cached
            return cached

    def _text_encoder_output_cache_is_safe(self) -> bool:
            """Enable output reuse only for a frozen, dropout-free encoder."""
            cache_safe = self._text_encoder_output_cache_safe
            if cache_safe is None:
                if any(
                    param.requires_grad for param in self.text_encoder.parameters()
                ):
                    # Do not cache False: some setup paths freeze the encoder
                    # after constructing this module but before training starts.
                    return False
                if any(
                    isinstance(module, torch.nn.modules.dropout._DropoutNd)
                    and module.p != 0
                    for module in self.text_encoder.modules()
                ):
                    return False
                configs = (
                    getattr(self.text_encoder, "config", None),
                    getattr(
                        getattr(self.text_encoder, "config", None),
                        "text_config",
                        None,
                    ),
                )
                for encoder_config in configs:
                    if encoder_config is None:
                        continue
                    for name in ("attention_dropout", "hidden_dropout", "dropout"):
                        value = getattr(encoder_config, name, 0.0)
                        if isinstance(value, (int, float)) and float(value) != 0.0:
                            return False
                cache_safe = True
                self._text_encoder_output_cache_safe = True
            # eval() is a cache precondition. With all dropout probabilities zero,
            # this changes no arithmetic but prevents a later parent train() call
            # from making cached and uncached executions differ.
            if cache_safe and self.text_encoder.training:
                self.text_encoder.eval()
            return cache_safe

    def invalidate_text_encoder_output_cache_safety(self) -> None:
            """Recheck cache safety after changing encoder trainability/config."""
            self._text_encoder_output_cache_safe = None
            self._prompt_embeds_cache.clear()

    def _get_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            *,
            device: torch.device,
            dtype: torch.dtype,
            max_sequence_length: int,
        ) -> torch.Tensor:
            prompt_list = [prompt] if isinstance(prompt, str) else prompt
    
            cache_output = self._text_encoder_output_cache_is_safe()
            output_key = (
                tuple(prompt_list),
                str(torch.device(device)),
                dtype,
                max_sequence_length,
            )
            if cache_output:
                cached_output = self._prompt_embeds_cache.get(output_key)
                if cached_output is not None:
                    return cached_output
    
            input_ids_batch = []
            for sample_idx in range(len(prompt_list)):
                token_key = (prompt_list[sample_idx], max_sequence_length)
                cached_ids = self._token_ids_cache.get(token_key)
                if cached_ids is not None:
                    input_ids_batch.append(cached_ids)
                    continue
                conversations = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are a helpful assistant who will provide prompts to an image generator.",
                            }
                        ],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompt_list[sample_idx]}]},
                ]
    
                input_ids = self.tokenizer.apply_chat_template(
                    conversations,
                    tokenize=True,
                    add_generation_prompt=False,
                    add_vision_id=False,
                    max_length=max_sequence_length,
                    truncation=True,
                    padding="max_length",
                )
                input_ids_tensor = torch.LongTensor(input_ids)
                if len(self._token_ids_cache) >= 4096:
                    self._token_ids_cache.clear()
                self._token_ids_cache[token_key] = input_ids_tensor
                input_ids_batch.append(input_ids_tensor)
    
            input_ids_batch = torch.stack(input_ids_batch, dim=0).to(device)
            outputs = self.text_encoder(input_ids_batch, output_hidden_states=True)
            hidden_states = outputs.hidden_states
    
            normalized_hidden_states = []
            for layer_idx in range(1, len(hidden_states)):
                hs = hidden_states[layer_idx]
                hs = (hs - hs.mean(dim=-1, keepdim=True)) / (hs.std(dim=-1, keepdim=True) + 1e-8)
                normalized_hidden_states.append(hs)
    
            prompt_embeds = torch.cat(normalized_hidden_states, dim=-1).to(dtype=dtype, device=device)
            if cache_output:
                # Bound GPU memory: this tensor is large because it concatenates
                # all text-encoder hidden layers. Cache only the latest exact batch.
                self._prompt_embeds_cache.clear()
                self._prompt_embeds_cache[output_key] = prompt_embeds.detach()
            return prompt_embeds

    def _denorm_latents(self, latents: torch.Tensor) -> torch.Tensor:
            """
            Inverse of normalization done in `prepare_latents`:
                latents_norm = (latents_raw - mean) / std
            so:
                latents_raw = latents_norm * std + mean
            Supports mean/std with or without a temporal dimension.
            """
            mean, std = self._get_latent_stats(
                device=latents.device, dtype=latents.dtype
            )
    
            # If mean/std have a temporal dimension, slice to match
            if mean.ndim == 5 and latents.ndim == 5 and mean.shape[2] >= latents.shape[2]:
                mean = mean[:, :, : latents.shape[2]]
            if std.ndim == 5 and latents.ndim == 5 and std.shape[2] >= latents.shape[2]:
                std = std[:, :, : latents.shape[2]]
    
            return latents * std + mean

    def _encode_video_to_latents_norm(self, video_bcthw: torch.Tensor) -> torch.Tensor:
            """
            Encode pixel video (B,3,T,H,W) into normalized latents (B,C,T_lat,H_lat,W_lat).
            Deterministic: uses encoder mean.
            """
            if video_bcthw.ndim != 5 or video_bcthw.shape[1] != 3:
                raise ValueError(f"video_bcthw must be (B,3,T,H,W), got {tuple(video_bcthw.shape)}")
            vb = video_bcthw.to(device=self.device, dtype=self.vae.dtype)
            enc = self.vae.encode(vb)
            if hasattr(enc, "latent_dist"):
                latents_raw = enc.latent_dist.mean
            else:
                latents_raw = enc
    
            latents_raw = latents_raw.to(dtype=torch.float32)
            mean, std = self._get_latent_stats(
                device=latents_raw.device, dtype=latents_raw.dtype
            )
    
            if mean.ndim == 5 and mean.shape[2] >= latents_raw.shape[2]:
                mean = mean[:, :, : latents_raw.shape[2]]
            if std.ndim == 5 and std.shape[2] >= latents_raw.shape[2]:
                std = std[:, :, : latents_raw.shape[2]]
    
            return (latents_raw - mean) / std


def extractor_cache_wrapper(original, options):
    from .cosmos_attention import install_cosmos_rope_cache

    max_entries = int(options.get("rope_cache_max_entries", 4))
    if max_entries < 1:
        raise ValueError("rope_cache_max_entries must be >= 1")

    class CachedExtractor(_Caches, original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not install_cosmos_rope_cache(self.transformer, max_entries):
                raise RuntimeError("Expected CosmosRotaryPosEmbed")

    CachedExtractor.__name__ = original.__name__
    return CachedExtractor
