# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from turbo_physai.optimizations.models.openwam.settings import value

"""Offline VAE latent store for RoboTwin training (two-step workflow).

Step 1, ``scripts/cache_vae_latents.py``: every rank encodes its share of the
training episodes with the frozen Wan VAE and writes one ``.npy`` per episode,
holding the latents of all its training windows in window order.
Step 2, training with vae_latent_cache=1: the readers return the
stored latent instead of decoding JPEG frames, and the VAE stays on the CPU.

Latents are the deterministic VAE mean, encoded by the same preprocessing and
batch_encode path as online training. ``meta.json`` records every reader
setting that changes the pixels; training refuses a store that differs.
"""

import hashlib
import json
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)
VERSION = 1
META = "meta.json"
# Marks the TI2V first-frame reference as latent[0] of the stored latents.
FIRST_FRAME_FROM_LATENTS = "first_frame_from_latents"
_OPEN_MAPS = 64


def directory():
    path = value("vae_latent_cache_dir", "").strip()
    if not path:
        raise ValueError("vae_latent_cache_dir must name the latent store directory")
    return path


def readers(dataset):
    from openwam.dataloader.robotwin import MultiTaskRoboTwinDataset, RoboTwinDataset

    if type(dataset) is MultiTaskRoboTwinDataset:
        result = list(dataset._sub_datasets)
    elif type(dataset) is RoboTwinDataset:
        result = [dataset]
    else:
        raise ValueError(
            "The VAE latent store currently supports RoboTwin datasets only"
        )
    for reader in result:
        # A stored latent identifies the pixels only for deterministic video.
        if reader._color_jitter is not None or reader.split != "train":
            raise ValueError(
                "The VAE latent store requires the train split without random video augmentation"
            )
    return result


def reader_key(reader):
    real = os.path.realpath(reader.data_root)
    tail = "__".join(part for part in real.replace("\\", "/").split("/")[-3:] if part)
    return f"{tail}__{hashlib.sha1(real.encode()).hexdigest()[:10]}"


def reader_signature(reader):
    return {
        "data_root": os.path.realpath(reader.data_root),
        "episodes": [os.path.basename(path) for path in reader._episode_files],
        "episode_lengths": [int(length) for length in reader._episode_lengths],
        "height": int(reader.height),
        "width": int(reader.width),
        "num_frames": int(reader.num_frames),
        "video_stride": int(reader.video_stride),
        "window_stride": int(reader.window_stride),
        "multiview": bool(reader.multiview),
        "cameras": list(reader.cameras),
        "target_camera": reader.target_camera,
    }


def episode_starts(reader, ep_idx):
    """Training window starts of one episode, in the reader's enumeration order."""
    length = reader._episode_lengths[ep_idx]
    if length < 2:
        return []
    return list(range(0, max(0, length - 2) + 1, reader.window_stride))


def check_windows(reader):
    """The store's row layout must reproduce the reader's own window index."""
    expected = [
        (ep_idx, start)
        for ep_idx in range(len(reader._episode_files))
        for start in episode_starts(reader, ep_idx)
    ]
    if expected != list(reader._window_index):
        raise ValueError(
            f"Window enumeration of {reader.data_root} differs from the latent store layout"
        )


def episode_path(root, reader, ep_idx, store_key=None):
    stem = os.path.splitext(os.path.basename(reader._episode_files[ep_idx]))[0]
    return os.path.join(
        root, store_key if store_key is not None else reader_key(reader), f"{stem}.npy"
    )


def validate_backbone(architecture):
    from openwam.model.video_backbone.wan.variants.ti2v import TI2VVariant
    from turbo_physai.optimizations.models.openwam.text import frozen_eval

    bb = architecture.video_backbone
    if (
        bb.video_encoder is not None
        or getattr(bb, "vace", None) is not None
        or type(getattr(bb, "_variant", None)) is not TI2VVariant
        or bb.vae is None
    ):
        raise ValueError(
            "The VAE latent store currently requires native Wan TI2V with the Wan VAE"
        )
    if not frozen_eval(bb.vae):
        raise ValueError("The VAE latent store requires a frozen VAE in eval mode")


def vae_fingerprint(vae):
    """Identify the VAE weights independently of checkpoint paths; computed on CPU copies."""
    digest = hashlib.sha1()
    for name, value in sorted(vae.state_dict().items()):
        value = value.detach().to(device="cpu", dtype=torch.float64)
        digest.update(
            f"{name}|{tuple(value.shape)}|{value.sum().item():.6e}|{value.abs().sum().item():.6e}".encode()
        )
    return digest.hexdigest()


def store_signature(dataset, vae):
    return {
        "version": VERSION,
        "dtype": "bfloat16",
        "vae": vae,
        "readers": {
            reader_key(reader): reader_signature(reader) for reader in readers(dataset)
        },
    }


def validate_signature(meta, expected):
    """Resolve runtime readers to stored keys, allowing dataset-root relocation.

    Legacy directory hashes locate files; they are not dataset identity. Match
    task/variant/data and every recorded preprocessing/episode field instead.
    Metadata does not fingerprint episode contents: replacing data in place
    still requires regenerating its cache.
    """
    for name in ("version", "dtype", "vae"):
        if meta.get(name) != expected[name]:
            raise ValueError(
                f"VAE latent store {name} mismatch: stored {meta.get(name)!r}, run {expected[name]!r}"
            )
    stored = meta.get("readers", {})
    resolved = {}

    def identity(reader):
        return tuple(reader["data_root"].replace("\\", "/").rstrip("/").split("/")[-3:])

    for key, reader in expected["readers"].items():
        candidates = (
            [key]
            if key in stored
            else [
                k for k, value in stored.items() if identity(value) == identity(reader)
            ]
        )
        if not candidates:
            raise ValueError(
                f"VAE latent store has no reader for task {'/'.join(identity(reader))}"
            )
        matches = [
            k
            for k in candidates
            if {n: v for n, v in stored[k].items() if n != "data_root"}
            == {n: v for n, v in reader.items() if n != "data_root"}
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous VAE latent readers for {reader['data_root']}: {matches}"
            )
        selected = matches[0] if matches else candidates[0]
        previous = stored[selected]
        changes = []
        for name in sorted((set(reader) | set(previous)) - {"data_root"}):
            value = reader.get(name)
            if previous.get(name) != value:
                old, new = repr(previous.get(name)), repr(value)
                changes.append(f"{name}: stored {old[:160]}, run {new[:160]}")
        if changes:
            raise ValueError(
                f"VAE latent store reader mismatch for {reader['data_root']}: "
                + "; ".join(changes)
            )
        if not matches:
            raise ValueError(
                f"VAE latent store reader fields differ for {reader['data_root']}"
            )
        if (
            selected in {"", ".", ".."}
            or "/" in selected
            or "\\" in selected
            or ":" in selected
        ):
            raise ValueError(f"Invalid latent reader directory key: {selected!r}")
        resolved[key] = selected
    return resolved


class EpisodeLatents:
    """Per-worker lazy memory maps of the episode files of one reader."""

    def __init__(self, root, reader, store_key=None):
        self.root, self.reader, self.maps = root, reader, {}
        self.store_key = store_key

    def __getstate__(self):
        # DataLoader workers reopen their own maps instead of pickling them.
        return {
            "root": self.root,
            "reader": self.reader,
            "maps": {},
            "store_key": self.store_key,
        }

    def __call__(self, ep_idx, start):
        array = self.maps.pop(ep_idx, None)
        if array is None:
            array = np.load(
                episode_path(self.root, self.reader, ep_idx, self.store_key),
                mmap_mode="r",
            )
            if len(self.maps) >= _OPEN_MAPS:
                # Each map holds a file descriptor; keep the recently used ones.
                self.maps.pop(next(iter(self.maps)))
        self.maps[ep_idx] = array
        row, remainder = divmod(start, self.reader.window_stride)
        if remainder or row >= array.shape[0]:
            raise IndexError(f"No stored latent for episode {ep_idx} start {start}")
        return torch.from_numpy(np.array(array[row])).view(torch.bfloat16)


def attach(dataset, vae):
    """Validate the store against this run and switch every reader to it."""
    root = directory()
    meta_path = os.path.join(root, META)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"VAE latent store has no {META}: run scripts/cache_vae_latents.py first ({root})"
        )
    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    expected = store_signature(dataset, vae)
    keys = validate_signature(meta, expected)
    shape = None
    for reader in readers(dataset):
        key = keys[reader_key(reader)]
        check_windows(reader)
        for ep_idx in range(len(reader._episode_files)):
            starts = episode_starts(reader, ep_idx)
            if not starts:
                continue
            path = episode_path(root, reader, ep_idx, key)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"VAE latent store is incomplete, missing {path}"
                )
            array = np.load(path, mmap_mode="r")
            if array.shape[0] != len(starts) or (
                shape is not None and array.shape[1:] != shape
            ):
                raise ValueError(
                    f"VAE latent store file {path} has shape {array.shape}, expected {len(starts)} rows"
                )
            shape = array.shape[1:]
        reader._latent_store = EpisodeLatents(root, reader, key)


def stacked(samples):
    """Batch latents from the samples, or None on the pixel path."""
    latents = [sample.get("video_latents") for sample in samples]
    if latents[0] is None:
        return None
    if any(latent is None for latent in latents):
        raise ValueError("Mixed stored-latent and pixel samples in one batch")
    return torch.stack(latents)
