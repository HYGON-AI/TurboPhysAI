# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot as _source
from ._source_binding import bind_missing_globals
from collections import OrderedDict
from threading import Lock
from typing import Any
import os as _os
from cosmos_framework.utils import log
from .video_decode import create_video_decoder
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import _LRU_VIDEO_CACHE_MAX_SIZE, _LRU_DATASET_MAX_LOADED
_decoder_cache_pid = None

class _LRUVideoDecoderCache:
    """Drop-in replacement for ``lerobot.datasets.video_utils.VideoDecoderCache``
    with LRU eviction.  When the cache exceeds *max_size* entries the
    least-recently-used decoder (and its file handle) is evicted.
    """

    def __init__(self, max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
        if max_size < 1:
            raise ValueError("decoder cache max_size must be positive")
        self._max_size = max_size
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get_decoder(self, video_path: str) -> Any:
        import fsspec

        video_path = str(video_path)

        with self._lock:
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
                self._hits += 1
                return self._cache[video_path][0]

            self._misses += 1
            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = create_video_decoder(file_handle, seek_mode="approximate")
            except Exception:
                file_handle.close()
                raise
            self._cache[video_path] = (decoder, file_handle)

            evicted = 0
            while len(self._cache) > self._max_size:
                _, (_, old_fh) = self._cache.popitem(last=False)
                try:
                    old_fh.close()
                except Exception:
                    pass
                evicted += 1
            self._evictions += evicted

            return decoder

    def clear(self) -> None:
        with self._lock:
            for _, file_handle in self._cache.values():
                try:
                    file_handle.close()
                except Exception:
                    pass
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

def _patch_decoder_cache(max_size: int = _LRU_VIDEO_CACHE_MAX_SIZE) -> None:
    """Replace the module-level ``_default_decoder_cache`` in LeRobot with an
    LRU-capped version to prevent unbounded memory growth in workers."""
    global _decoder_cache_pid
    import lerobot.datasets.video_utils as _vu

    pid = _os.getpid()
    if _decoder_cache_pid == pid and isinstance(_vu._default_decoder_cache, _LRUVideoDecoderCache):
        return

    # Spawn imports a fresh module; fork inherits a cache owned by another PID.
    # In either case the decoder/file-handle cache must belong to this process.
    if isinstance(_vu._default_decoder_cache, _LRUVideoDecoderCache):
        _vu._default_decoder_cache.clear()
    lru_cache = _LRUVideoDecoderCache(max_size=max_size)
    _vu._default_decoder_cache = lru_cache
    _decoder_cache_pid = pid
    log.info(f"Action decoder LRU ready: pid={pid}, max_size={max_size}")

def __init__(
    self,
    *,
    fps: float,
    chunk_length: int,
    split_seed: int,
    split_val_ratio: float,
    split: str,
    mode: str,
    embodiment_type: str,
    viewpoint: Viewpoint,
    pose_convention: str | None = None,
    rotation_format: str | None = None,
    action_normalization: ActionNormalization | None = None,
    tolerance_s: float = 1e-4,
    max_loaded_datasets: int = _LRU_DATASET_MAX_LOADED,
    skip_video_loading: bool = False,
    video_backend: str | None = None,
    sample_stride: int = 1,
    enable_fast_init: bool = False,
    fast_init_max_workers: int = 64,
    min_episode_length_frames: int | None = None,
) -> None:
    super(_source.BaseActionLeRobotDataset, self).__init__()
    _ensure_hf_hub_offline()
    _patch_decoder_cache()
    self._memprofile = _memprofile_enabled()

    assert sample_stride >= 1, f"sample_stride must be >= 1, got {sample_stride}"
    assert fast_init_max_workers >= 1, f"fast_init_max_workers must be >= 1, got {fast_init_max_workers}"
    assert action_normalization is None or action_normalization in _ACTION_NORMALIZATION_CHOICES, (
        f"action_normalization must be None or one of {_ACTION_NORMALIZATION_CHOICES}, got {action_normalization!r}"
    )

    with rss_tracker(f"{self.__class__.__name__}.__init__", enabled=self._memprofile):
        self._fps = fps
        self._dt = 1.0 / fps
        self._chunk_length = chunk_length
        self._split_seed = split_seed
        self._split_val_ratio = split_val_ratio
        self._split = _normalize_split(split)
        self._mode = mode
        self._embodiment_type = embodiment_type
        self._viewpoint: Viewpoint = viewpoint
        self._pose_convention = pose_convention
        self._rotation_format = rotation_format
        self._action_normalizer: ActionNormalizer | None = None
        if action_normalization is not None:
            self._action_normalizer = resolve_action_normalization(
                action_normalization, self._load_norm_stats(action_normalization)
            )
        self._tolerance_s = tolerance_s
        self._max_loaded_datasets = max_loaded_datasets
        self._skip_video_loading = skip_video_loading
        self._video_backend = video_backend
        self._sample_stride = sample_stride
        self._enable_fast_init = enable_fast_init
        self._fast_init_max_workers = fast_init_max_workers
        # Optional post-filter on raw episode length. When set, episodes
        # whose raw frame count is below this threshold are dropped from
        # ``_episode_records`` after ``build_episode_spans`` runs. Lets
        # eval configs select e.g. only ``> 60s`` wall-clock episodes
        # (1800 raw frames at native 30 fps) while keeping ``chunk_length``
        # (the fetch window) at a smaller value such as 900 (60 s @ fps=15).
        # Subclasses that override ``_append_index_records`` are expected
        # to honor this attribute themselves.
        self._min_episode_length_frames: int | None = min_episode_length_frames
        self._delta_timestamps: dict[str, list[float]] = {}
        self._to_opencv: np.ndarray | dict[str, np.ndarray] = np.eye(3, dtype=np.float32)

        if pose_convention is None:
            log.warning(
                f"{self.__class__.__name__}: pose_convention is not set. "
                "Consider specifying 'backward_framewise' or 'backward_anchored'."
            )

        self._datasets: list[LeRobotDataset | None] = []
        self._dataset_build_args: list[dict[str, Any] | None] = []
        self._loaded_lru: OrderedDict[int, None] = OrderedDict()

        # -- Flat index structures (populated by _append_index_records) --
        # Together these two lists form a searchable map from a flat
        # global index to (dataset, row, episode, frame).  One entry per
        # episode span across *all* registered sources.
        #
        # _episode_records[i] = (ds_idx, sample_start, valid_len, episode_id)
        #   ds_idx       – which source dataset (index into _datasets)
        #   sample_start – first row of this span in that dataset's table
        #   valid_len    – number of usable frames in this span
        #   episode_id   – the episode this span belongs to
        #
        # _episode_cum_ends[i] = running total of valid_len through span i
        #   Used for O(log N) lookup via bisect_right in _resolve_index.
        self._episode_records: list[tuple[int, int, int, int]] = []
        self._episode_cum_ends: list[int] = []
        self._num_valid_indices = 0
        self._domain_id = get_domain_id(self._embodiment_type)

        # Deferred-init shard roots — a list of root paths.
        # Subclasses populate this in __init__; _register_sources()
        # reads _delta_timestamps and _tolerance_s from self (both
        # initialised above, with _delta_timestamps overridden by
        # each subclass).
        # ActionUnifiedIterableDataset.assign_worker uses len() for
        # round-robin shard distribution and _register_sources(indices)
        # for deferred loading.  When empty, shard distribution is
        # skipped (every worker iterates the full dataset).
        self._all_shard_roots: list[str] = []

def _register_sources(self, indices: list[int] | None = None) -> None:
    """Register a subset (or all) of the shard roots in ``_all_shard_roots``.

    Called by ``ActionUnifiedIterableDataset.assign_worker`` during training,
    or explicitly by eval/visualization scripts after construction.

    ``_all_shard_roots`` is a list of root paths.  Per-shard args that are
    shared across all shards (``delta_timestamps``, ``tolerance_s``) are
    taken from ``self``.  Subclasses may override this for extra per-shard
    setup (e.g. loading instruction segments).

    When ``enable_fast_init=True``, ``LeRobotDatasetMetadata`` (a pure-IO
    read of ``info.json`` + ``episodes.parquet`` + ``tasks.parquet``) is
    prefetched in a thread pool and handed to the order-sensitive
    serial register loop via ``prefetched_meta=``.  Shard count scales
    the speedup; for single-shard datasets the two paths are
    equivalent.

    Args:
        indices: Which entries of ``_all_shard_roots`` to register.
            ``None`` means all.
    """
    if indices is None:
        indices = list(range(len(self._all_shard_roots)))
    if not indices:
        return

    roots = [self._all_shard_roots[i] for i in indices]

    if self._enable_fast_init:
        # ``_ensure_hf_hub_offline`` already ran in ``__init__`` and is
        # idempotent; no need to re-invoke here.
        workers = max(1, min(self._fast_init_max_workers, len(roots)))
        metas: list[LeRobotDatasetMetadata | None] = _parallel_map(
            lambda root: LeRobotDatasetMetadata(repo_id="local", root=root, revision="local"),
            roots,
            max_workers=workers,
            label=f"{type(self).__name__}: LeRobotDatasetMetadata prefetch",
        )
    else:
        metas = [None] * len(roots)

    for root, meta in zip(roots, metas):
        label = root.rsplit("/", 1)[-1] if "/" in root else root
        self._register_source(
            root=root,
            delta_timestamps=self._delta_timestamps,
            tolerance_s=self._tolerance_s,
            video_backend=self._video_backend,
            dataset_label=label,
            prefetched_meta=meta,
        )

def _get_dataset(self, ds_idx: int) -> LeRobotDataset:
    """Get or lazily construct the LeRobot dataset for the given source index.

    Loaded datasets are tracked with LRU ordering.  When the number of
    loaded datasets exceeds ``_max_loaded_datasets`` the least-recently-used
    dataset is evicted (set back to ``None``) so the GC can reclaim it.
    """
    # __init__ runs only in the parent when a DataLoader uses spawn.
    # Install before both the cached and lazy dataset access paths.
    _patch_decoder_cache()
    ds = self._datasets[ds_idx]
    if ds is not None:
        self._loaded_lru.move_to_end(ds_idx)
        return ds

    _ensure_hf_hub_offline()

    build_args = self._dataset_build_args[ds_idx]
    if build_args is None:
        raise RuntimeError(f"Missing dataset build args for dataset index {ds_idx}")

    # Evict least-recently-used datasets before loading a new one.
    while len(self._loaded_lru) >= self._max_loaded_datasets:
        evict_idx, _ = self._loaded_lru.popitem(last=False)
        self._datasets[evict_idx] = None

    with rss_tracker(
        f"[WORKER {_os.getpid()}] Lazy-loaded ds[{ds_idx}]",
        enabled=self._memprofile,
        extras_fn=lambda: [f"total loaded={len(self._loaded_lru)}/{len(self._datasets)}"],
    ):
        delta_ts = build_args["delta_timestamps"]
        if self._skip_video_loading:
            # Covers both LeRobot v2 (``observation.images.<name>``) and
            # v3 (``observation.image.<name>``) video-column conventions.
            delta_ts = {k: v for k, v in delta_ts.items() if not k.startswith("observation.image")}

        log.info(f"Loading shard root={build_args['root']}")
        ds = LeRobotDataset(
            repo_id=build_args["repo_id"],
            root=build_args["root"],
            delta_timestamps=delta_ts,
            tolerance_s=build_args["tolerance_s"],
            force_cache_sync=build_args["force_cache_sync"],
            download_videos=build_args["download_videos"],
            video_backend=build_args["video_backend"],
            revision=build_args["revision"],
            episodes=None,
        )
        if self._skip_video_loading:
            ds.meta.info["features"] = {
                k: v for k, v in ds.meta.info["features"].items() if v.get("dtype") != "video"
            }
        self._datasets[ds_idx] = ds
        self._loaded_lru[ds_idx] = None

    return ds


bind_missing_globals(globals(), _source)
