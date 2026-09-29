# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""WanVAE / Wan2pt2VAEInterface HIP adaptations (adapted source port).

- ``WanVAE`` gains an ``is_amp`` toggle; the interface pins ``is_amp=False``
  (pure-bf16 forward, upstream numerical behavior) and converts Conv2d/Conv3d
  weights to channels-last on HIP.
- ``WanVAE_._encode_chunk_impl``/``encode``: NDHWC chunk layout assertions and
  channels-last chunk normalization for the eager and AOT paths.
- ``Wan2pt2VAEInterface.__init__``: 720p ``encode_chunk_frames`` 8 -> 12.
- ``compile_encode``: warmup-shape (H, W) unpack fix carried from the adapted
  source and simplified AOT cache-dir handling.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 as _vae_mod
from cosmos_framework.utils import log

from .._source_binding import bind_missing_globals
from .layout import _USE_CHANNELS_LAST, _is_layout_contiguous, _to_channels_last_3d


def wan_vae_post_init(original, options):
    """Wrap ``WanVAE.__init__``: keep the pure-dtype forward, add channels-last weights."""
    del options

    def initialized(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.is_amp = False
        self.context = nullcontext()
        if _USE_CHANNELS_LAST:
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    module.to(memory_format=torch.channels_last)
                elif isinstance(module, nn.Conv3d):
                    module.to(memory_format=torch.channels_last_3d)
            log.info("WanVAE: enabled channels-last Conv2d/Conv3d weights")
        return result

    return initialized


@torch.no_grad()
def wan_vae_encode(self, videos: torch.Tensor) -> torch.Tensor:
    """Encode a batch of videos.

    AOT-compiled chunk functions (if installed on ``self.model`` via
    :meth:`Wan2pt2VAEInterface.compile_encode`)
    are dispatched from inside ``WanVAE_.encode`` at the per-chunk level,
    preserving the chunked encoding loop and temporal padding logic.

    Args:
        videos: Tensor of shape ``[B, C, T, H, W]``.

    Returns:
        Tensor of shape ``[B, z_dim, T//4, H//16, W//16]``.
    """
    in_dtype = videos.dtype
    with self.context:
        if not self.is_amp:
            videos = videos.to(self.dtype)
        latent = self.model.encode(videos, self.scale)
    latent = latent.to(in_dtype)
    if _USE_CHANNELS_LAST:
        latent = latent.contiguous()
    return latent


@torch.no_grad()
def wan_vae_decode(self, zs: torch.Tensor, clear_decoder_cache: bool = True) -> torch.Tensor:
    """Decode a batch of latent tensors.

    Args:
        zs: Tensor of shape ``[B, z_dim, T, H, W]``.
        clear_decoder_cache: Whether to clear the decoder cache between decode calls.

    Returns:
        Tensor of shape ``[B, C, T, H, W]``.
    """
    in_dtype = zs.dtype
    with self.context:
        if not self.is_amp:
            zs = zs.to(self.dtype)
        zs = _to_channels_last_3d(zs)
        video_recon = self.model.decode(zs, self.scale, clear_decoder_cache)
    video_recon = video_recon.to(in_dtype)
    if _USE_CHANNELS_LAST:
        video_recon = video_recon.contiguous()
    return video_recon


def wan_vae_encode_chunk_impl(
    self,
    x_chunk: torch.Tensor,
    feat_cache: list[torch.Tensor | None],
    scale: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
    """Run the encoder on one temporal chunk and normalize the output.

    Defined as an instance method (not a closure inside ``encode``) so
    that ``_ChunkEncodeForAOT`` can wrap it for ``torch.export``.

    Note: Since ``feat_cache`` is mutated in-place by the encoder (each
    ``CausalConv3d`` layer overwrites its slot), we pass a shallow copy
    to preserve the original cache for compilation.
    """
    feat_cache = list(feat_cache)

    assert all(c is None or _is_layout_contiguous(c) for c in feat_cache)
    assert _is_layout_contiguous(x_chunk)

    out = self.encoder(x_chunk, feat_cache=feat_cache)

    assert _is_layout_contiguous(out)
    assert all(c is None or _is_layout_contiguous(c) for c in feat_cache)

    # Project encoder features through conv1, split to mu/log_var, and normalize.
    mu, _log_var = self.conv1(out).chunk(2, dim=1)
    return self._normalize_latent(mu, scale), feat_cache


def wan_vae_inner_encode(self, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Chunked causal encoding that converts pixel-space video to latent space.

    The Wan 2.2 VAE encoder uses causal 3D convolutions, which means each
    ``CausalConv3d`` layer needs temporal context from previous frames.  For
    long videos, running the full sequence at once would create intermediate
    tensors of shape ``(B*T, C, H, W)`` that can exceed Triton's int32
    indexing limit.  Instead, we split the video into fixed-size temporal
    chunks and maintain a per-layer feature cache (``feat_cache``) so each
    chunk can access the causal context from previous chunks.

    **Encoding strategy:**

    1. The first frame is encoded alone (the "key-frame prime") to seed
       the causal caches with initial state.
    2. Subsequent frames are processed in chunks of ``temporal_window`` (e.g.
       68 frames).  Each chunk reads from and writes to the shared
       ``feat_cache`` list, which stores the last ``CACHE_T=2`` frames of
       activations per ``CausalConv3d`` layer.
    3. The per-chunk latents are concatenated along the temporal axis.

    **AOT compilation:**

    When ``_aot_chunk_fns`` has been installed (via
    :meth:`Wan2pt2VAEInterface.compile_encode`),
    each chunk is dispatched to a pre-compiled ``.pt2`` function keyed by
    ``(T_chunk, H_patch, W_patch, cache_t)``.  Padding ensures every chunk
    (except possibly the last, handled by ``should_pad``) has exactly
    ``temporal_window`` frames, keeping the set of compiled input shapes small.

    Args:
        x: Pixel-space video tensor of shape ``[B, 3, T, H, W]``.
            ``T`` must satisfy ``T == 1`` or ``(T - 1) % 4 == 0`` (the
            4x temporal compression constraint).  Use ``pad_video_batch``
            to pad to a valid length before calling.
        scale: Tuple of ``(mean, inv_std)`` tensors, each of shape
            ``[z_dim]``.  Used to normalize the latent: ``(z - mean) * inv_std``.

    Returns:
        Normalized latent tensor of shape ``[B, z_dim, T_latent, H//16, W//16]``
        where ``T_latent = 1 + (T - 1) // 4``.  Always corresponds to the
        *original* (unpadded) input length, even when internal padding was applied.
    """
    T, H, W = x.shape[2], x.shape[3], x.shape[4]

    # ``temporal_window`` can be a per-resolution mapping (e.g.
    # {"256": 68, "480": 32, "720": 16}) from a Hydra/OmegaConf config,
    # which arrives as a ``DictConfig`` (not a plain ``dict``).
    # Using ``Mapping`` catches both.
    if isinstance(self.temporal_window, Mapping):
        resolution = get_vision_data_resolution((H, W))
        temporal_window = self.temporal_window[resolution]
    else:
        temporal_window = self.temporal_window

    assert T == 1 or (T - 1) % 4 == 0, (
        f"Input temporal length must be 4n+1 (got {T}). "
        "Use pad_video_batch to pad before encoding, check wan2pt2_vae_4x16x16_test on how to use it."
    )

    # The 4x temporal compression maps T pixel frames → ceil-like latent frames.
    # For T=1 (single image), latent_T=1.  For T=4n+1, latent_T=n+1.
    latent_T = 1 + (T - 1) // 4

    # Certain short-clip durations (e.g. robotics datasets with T=17) can be
    # encoded at their exact length, avoiding the overhead of padding to the
    # next chunk boundary.  All other lengths are padded so that each chunk
    # has exactly ``temporal_window`` frames, giving the compiled function a
    # fixed input shape per {resolution, aspect_ratio} bucket.
    should_pad = T not in self._encode_exact_durations

    if should_pad:
        # Pad T to ``1 + k * temporal_window`` so that after removing the 1-frame
        # prime, the remaining frames divide evenly into ``temporal_window``-sized chunks.
        T = 1 + ((T - 1 + temporal_window - 1) // temporal_window) * temporal_window
        x = F.pad(x, (0, 0, 0, 0, 0, T - x.shape[2]))

    # One cache slot per CausalConv3d layer in the encoder, initially all None.
    enc_cache = self._new_enc_cache()

    # Patchify merges each 2×2 spatial patch into the channel dim:
    # [B, 3, T, H, W] → [B, 12, T, H//2, W//2].
    x = patchify(x, patch_size=2)

    aot_chunk_fns: dict | None = getattr(self, "_aot_chunk_fns", None)
    H_patch, W_patch = x.shape[3], x.shape[4]

    def _run_chunk(
        x_chunk: torch.Tensor,
        feat_cache: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Encode one chunk, through the AOT path or eager fallback.

        If AOT-compiled chunk functions are installed, the function is
        looked up by ``(T_chunk, H_patch, W_patch, cache_t)`` where
        ``cache_t`` is the minimum temporal extent of the cache tensors
        (0 = all-None prime, 1 = post-prime, 2 = steady state).

        Both full-size and remainder chunks (from ``encode_exact_durations``)
        are dispatched through the AOT path when a matching compiled
        function exists; uncompiled shapes fall back to eager.
        """
        is_prime = feat_cache[0] is None

        # Ensure the layout expected by the selected backend. HIP keeps chunks NDHWC so
        # Conv3d does not pay a layout conversion at every layer.
        memory_format = torch.channels_last_3d if _USE_CHANNELS_LAST else torch.contiguous_format
        x_chunk = x_chunk.contiguous(memory_format=memory_format)

        if aot_chunk_fns is not None:
            cache_t = 0 if is_prime else feat_cache[0].shape[2]
            aot_key = (x_chunk.shape[2], H_patch, W_patch, cache_t)
            aot_fn = aot_chunk_fns.get(aot_key)

            if aot_fn is not None:
                return aot_fn(x_chunk, feat_cache)

        return self._encode_chunk_impl(x_chunk, feat_cache, scale)

    # --- Chunked encoding loop ---
    # Chunk 0: single-frame "key-frame prime" to seed all causal caches.
    out, enc_cache = _run_chunk(x[:, :, :1], feat_cache=enc_cache)
    outs = [out]

    # Chunks 1..N: process the remaining frames in fixed-size windows.
    for start in range(1, T, temporal_window):
        x_chunk = x[:, :, start : start + temporal_window]
        out, enc_cache = _run_chunk(x_chunk, feat_cache=enc_cache)
        outs.append(out)

    final_out = torch.cat(outs, dim=2) if len(outs) > 1 else outs[0]

    # If we padded the input, trim the latent back to the original length.
    if should_pad:
        final_out = final_out[:, :, :latent_T]
    return final_out


def interface_chunk_frames(original, options):
    """Wrap ``Wan2pt2VAEInterface.__init__``: raise the 720p chunk to 12 frames."""
    del options

    def initialized(self, *args, **kwargs):
        encode_chunk_frames = kwargs.get("encode_chunk_frames", 4)
        if isinstance(encode_chunk_frames, int):
            kwargs["encode_chunk_frames"] = {"256": 68, "480": 24, "720": 12}
        return original(self, *args, **kwargs)

    return initialized


def _collect_warmup_shapes(
    tokenizer: "Wan2pt2VAEInterface",
    warmup_resolutions: Sequence[str],
    aspect_ratio: str | None = None,
) -> list[_ShapeInfo]:
    """Return ``[(chunk_frames, H_patch, W_patch), ...]`` for all warmup shapes.

    Expands *warmup_resolutions* into concrete spatial shapes using
    ``VIDEO_RES_SIZE_INFO``.  Each resolution may have multiple aspect
    ratios (e.g. ``"16,9"``, ``"9,16"``, ``"1,1"``); optionally filtered
    to a single ratio via *aspect_ratio*.  ``chunk_frames`` is looked up
    from the tokenizer (scalar or per-resolution dict).  Spatial
    dimensions are halved to account for patchify (``patch_size=2``).
    """
    all_shapes: list[_ShapeInfo] = []
    for res_key in warmup_resolutions:
        if res_key not in VIDEO_RES_SIZE_INFO:
            raise ValueError(f"Resolution {res_key} not found in VIDEO_RES_SIZE_INFO")

        if isinstance(tokenizer.encode_chunk_frames, Mapping):
            if res_key not in tokenizer.encode_chunk_frames:
                raise ValueError(f"Resolution {res_key} not found in tokenizer.encode_chunk_frames")

        res_dict = VIDEO_RES_SIZE_INFO[res_key]
        if aspect_ratio is not None:
            if aspect_ratio not in res_dict:
                raise ValueError(f"Aspect ratio {aspect_ratio} not found in resolution {res_key}")
            res_dict = {aspect_ratio: res_dict[aspect_ratio]}

        for H, W in res_dict.values():
            if isinstance(tokenizer.encode_chunk_frames, Mapping):
                chunk_frames = tokenizer.encode_chunk_frames[res_key]
            else:
                chunk_frames = tokenizer.encode_chunk_frames

            H_patch, W_patch = H // 2, W // 2
            all_shapes.append((chunk_frames, H_patch, W_patch))
    return all_shapes


@torch.no_grad()
def interface_compile_encode(
    self,
    warmup_resolutions: Sequence[str],
    output_dir: str,
    aspect_ratio: str | None = None,
    # ignores torch compile args
    **kwargs,
) -> None:
    """AOT-compile the tokenizer's chunk-level encode for every resolution.

    Compiles ``WanVAE_._encode_chunk_impl`` for each
    ``(resolution, aspect_ratio, cache_t)`` variant, producing ``.pt2``
    packages that are loaded on all ranks for zero-overhead dispatch
    during training.

    **Variant enumeration** — for each resolution, three standard
    ``cache_t`` variants (prime, post-prime, steady-state) are compiled.
    When ``encode_exact_durations`` is configured, additional remainder
    variants are appended.

    **Distribution** — individual variants are assigned round-robin
    across ranks.  Reference caches are built lazily.

    **Shared weights** — packages are compiled with
    ``package_constants_in_so=False``.  After loading, each runner
    receives the same encoder weights via
    ``load_constants(user_managed=True)``.

    Compiled functions are installed as ``self.model.model._aot_chunk_fns``
    for dispatch by ``_run_chunk`` inside ``WanVAE_.encode``.

    Args:
        warmup_resolutions: Resolution keys (e.g. ``["256", "480", "720"]``).
        output_dir: Root directory under which compiled ``.pt2`` packages
            are written (an ``aot_tokenizer/`` subdirectory will be
            created).  Typically the job's local output path
            (``config.job.path_local``).
        aspect_ratio: If given, only compile this single aspect ratio per
            resolution instead of all available ratios.
    """
    import torch._inductor
    import torch.distributed as dist

    log.info(f"AOT chunk-level warmup for resolutions: {warmup_resolutions}", rank0_only=False)
    start_time = time.time()

    save_dir = os.path.join(output_dir, "aot_tokenizer")

    all_shapes = _collect_warmup_shapes(self, warmup_resolutions, aspect_ratio)

    is_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    wanvae = self.model  # WanVAE (plain class)
    wanvae_model = wanvae.model  # WanVAE_ (nn.Module)
    scale = wanvae.scale  # (mean, 1/std)
    n_cache_slots = wanvae_model._enc_conv_num

    if rank == 0:
        log.info(f"Saving AOT compiled packages to {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    if is_distributed:
        dist.barrier()

    # -- Helper functions --------------------------------------------------

    def _rand_cache(cache: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
        return [torch.rand_like(c) if c is not None else None for c in cache]

    def _rand_input(t: int, h: int, w: int) -> torch.Tensor:
        return _to_channels_last_3d(torch.rand((1, 12, t, h, w), dtype=torch.bfloat16, device="cuda"))

    def _compile_variant(
        wrapper: _ChunkEncodeForAOT,
        aot_key: _AOTChunkKey,
        ref_cache: list[torch.Tensor | None],
    ) -> str | None:
        """Export + compile one variant, returning the .pt2 path or None."""
        t_chunk, H_patch, W_patch, cache_t = aot_key
        pkg_name = f"chunk_ct{cache_t}_{t_chunk}f_{H_patch}x{W_patch}.pt2"
        pkg_path = os.path.join(save_dir, pkg_name)

        if os.path.exists(pkg_path):
            log.info(f"Rank {rank}: reusing cached {pkg_name}", rank0_only=False)
            return pkg_path

        t0 = time.time()
        try:
            exported = torch.export.export(
                wrapper,
                (_rand_input(t_chunk, H_patch, W_patch), _rand_cache(ref_cache)),
                strict=False,
            )
            torch._inductor.aoti_compile_and_package(
                exported,
                package_path=pkg_path,
                inductor_configs={"aot_inductor.package_constants_in_so": False},
            )
            log.info(
                f"Rank {rank}: AOT compiled cache_t={cache_t} "
                f"{t_chunk}f {H_patch}x{W_patch} in {time.time() - t0:.1f}s",
                rank0_only=False,
            )
            return pkg_path
        except Exception as e:
            log.warning(
                f"Rank {rank}: AOT compile failed for cache_t={cache_t} {t_chunk}f {H_patch}x{W_patch}: {e}",
                rank0_only=False,
            )
            return None

    # -- Enumerate all variant keys and distribute across ranks ------------

    all_variant_keys: list[tuple[_AOTChunkKey, _ShapeInfo]] = []
    seen_keys: set[_AOTChunkKey] = set()
    for chunk_frames, H_patch, W_patch in all_shapes:
        for cache_t in (0, 1, 2):
            t_chunk = 1 if cache_t == 0 else chunk_frames
            aot_key: _AOTChunkKey = (t_chunk, H_patch, W_patch, cache_t)

            assert aot_key not in seen_keys, f"Duplicate AOT key: {aot_key}"
            seen_keys.add(aot_key)
            all_variant_keys.append((aot_key, (chunk_frames, H_patch, W_patch)))

        for T in sorted(self.encode_exact_durations or []):
            remaining = T - 1
            if remaining <= 0:
                continue
            remainder = remaining % chunk_frames
            if remainder == 0:
                continue
            n_full = remaining // chunk_frames
            cache_t = 1 if n_full == 0 else 2
            aot_key = (remainder, H_patch, W_patch, cache_t)

            if aot_key not in seen_keys:
                seen_keys.add(aot_key)
                all_variant_keys.append((aot_key, (chunk_frames, H_patch, W_patch)))

    my_variant_keys = [v for i, v in enumerate(all_variant_keys) if i % world_size == rank]
    log.info(
        f"Rank {rank}: assigned {len(my_variant_keys)}/{len(all_variant_keys)} variants (world_size={world_size})",
        rank0_only=False,
    )

    # -- Build reference caches lazily, only for this rank's shapes --------

    wrapper = _ChunkEncodeForAOT(wanvae_model, scale[0], scale[1])
    wrapper.eval()

    def _get_ref_caches(
        chunk_frames: int,
        H_patch: int,
        W_patch: int,
    ) -> dict[int, list[torch.Tensor | None]]:
        cache_ct0: list[torch.Tensor | None] = [None] * n_cache_slots
        _, cache_ct1 = wanvae_model._encode_chunk_impl(
            _rand_input(1, H_patch, W_patch),
            list(cache_ct0),
            scale,
        )
        _, cache_ct2 = wanvae_model._encode_chunk_impl(
            _rand_input(chunk_frames, H_patch, W_patch),
            list(cache_ct1),
            scale,
        )
        return {0: cache_ct0, 1: cache_ct1, 2: cache_ct2}

    ref_cache_map: dict[_ShapeInfo, dict[int, list[torch.Tensor | None]]] = {}

    my_pkg_paths: dict[_AOTChunkKey, str] = {}
    for aot_key, shape_info in my_variant_keys:
        cache_t = aot_key[3]
        if shape_info not in ref_cache_map:
            ref_cache_map[shape_info] = _get_ref_caches(*shape_info)
        ref_cache = ref_cache_map[shape_info][cache_t]
        pkg_path = _compile_variant(wrapper, aot_key, ref_cache)
        if pkg_path is not None:
            my_pkg_paths[aot_key] = pkg_path

    # -- Gather .pt2 paths from every rank so all ranks can load all variants.
    if is_distributed:
        gathered: list[dict[_AOTChunkKey, str] | None] = [None] * world_size
        dist.all_gather_object(gathered, my_pkg_paths)
        pkg_paths: dict[_AOTChunkKey, str] = {}
        for rank_paths in gathered:
            if rank_paths:
                pkg_paths.update(rank_paths)
        dist.barrier()
    else:
        pkg_paths = my_pkg_paths

    # -- Load every .pt2 package and bind to the existing encoder weights. --
    device_index = torch.cuda.current_device()
    state_dict = wrapper.state_dict()

    loaded_fns: dict[_AOTChunkKey, Callable] = {}
    for key, pkg_path in pkg_paths.items():
        try:
            fn = torch._inductor.aoti_load_package(pkg_path, device_index=device_index)

            required_keys = set(fn.get_constant_fqns())
            constants_map = {k: v for k, v in state_dict.items() if k in required_keys}
            fn.load_constants(constants_map, check_full_update=True, user_managed=True)

            loaded_fns[key] = fn
        except Exception as e:
            log.warning(
                f"Rank {rank}: failed to load {pkg_path}: {e}",
                rank0_only=False,
            )

    wanvae_model._aot_chunk_fns = loaded_fns

    log.info(
        f"Rank {rank}: AOT compiled {len(my_pkg_paths)}, "
        f"loaded {len(loaded_fns)}/{len(all_variant_keys)} chunk variants, "
        f"time: {time.time() - start_time:.2f}s",
        rank0_only=False,
    )

    # Clean up .pt2 files so stale packages don't persist across restarts.
    if is_distributed:
        dist.barrier()
    if rank == 0:
        import shutil

        try:
            shutil.rmtree(save_dir)
            log.info(f"Cleaned up AOT cache dir: {save_dir}")
        except OSError as e:
            log.warning(f"Failed to clean AOT cache dir {save_dir}: {e}")

    if not loaded_fns:
        raise RuntimeError("AOT compilation produced no loadable functions")


bind_missing_globals(globals(), _vae_mod)
