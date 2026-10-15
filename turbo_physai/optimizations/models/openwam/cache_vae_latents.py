# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""Step 1 of the VAE latent store: encode every RoboTwin training window once.

Use the same dataloader/model overrides and Turbo Group options as training.
Set openwam.data.options.vae_latent_cache_dir in the OptimizationConfig, then
run through Turbo from the unmodified OpenWAM repository::

    turbo-physai run --optimization-config /path/to/cache-config.yaml \\
        torchrun --nproc_per_node=8 --module \\
        turbo_physai.optimizations.models.openwam.cache_vae_latents \\
        dataloader=robotwin training.finetune_ckpt_path=...

Each rank encodes a disjoint set of episodes and writes one ``.npy`` per
episode (atomic rename), so an interrupted run resumes by skipping finished
files. Step 2 is the normal training command with
vae_latent_cache=1 and the same vae_latent_cache_dir.

With text_cache=1, also encode all possible training prompts with
the frozen T5. text_cache_dir defaults to the latent directory's
``text`` subdirectory. Step 2 uses the same flag/directory and keeps T5 on CPU.
Rerunning with complete v2 text files builds ``text/index.json`` without T5
encoding. Rank lengths are gathered on CPU and the complete index is published
atomically, so training need not scan the individual text files at startup.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path.cwd()
logger = logging.getLogger(__name__)


class _Episodes(torch.utils.data.Dataset):
    """All frames of one episode, decoded and assembled once (uint8 [L, H, W, 3]).

    Consecutive windows overlap in all but one frame; decoding per window would
    repeat every JPEG decode and multiview assembly about num_frames times.
    """

    def __init__(self, readers, items):
        self.readers, self.items = readers, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        r, e = self.items[index]
        return r, e, torch.from_numpy(_episode_frames(self.readers[r], e))


def _episode_frames(reader, ep_idx):
    """The frames RoboTwinDataset._build_sample would sample from, for the whole episode."""
    import h5py

    from openwam.dataloader.transforms.multiview import crop_and_resize

    length = reader._episode_lengths[ep_idx]
    with h5py.File(reader._episode_files[ep_idx], "r") as f:
        if reader.multiview:
            frames = reader._read_multiview_frames(f, reader.cameras, 0, length)
        else:
            frames = [
                crop_and_resize(frame, reader.height, reader.width)
                for frame in reader._read_camera_frames(
                    f, reader.target_camera, 0, length
                )
            ]
    if any(frame.mode != "RGB" for frame in frames):
        raise ValueError(f"Expected RGB frames in {reader._episode_files[ep_idx]}")
    return np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])


def _window_pixels(frames, reader, start, dtype, min_value=-1, max_value=1):
    """Same tensor as wan_encode.preprocess_video(_build_sample(...)["video"]).

    A window past the episode end repeats the last frame, as _build_sample pads.
    The arithmetic matches preprocess_video's uint8 path; its float32 path gives
    identical values because 0..255 are exact in BF16.
    """
    last = frames.shape[0] - 1
    index = torch.tensor(
        [min(start + i, last) for i in reader._video_sample_indices],
        device=frames.device,
    )
    pixels = frames.index_select(0, index).to(dtype=dtype)
    pixels = pixels * ((max_value - min_value) / 255) + min_value
    return pixels.permute(3, 0, 1, 2).unsqueeze(0)


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp{os.environ.get('RANK', '0')}"
    with open(temporary, "wb") as handle:
        np.save(handle, torch.stack(rows).view(torch.int16).numpy())
    os.replace(temporary, path)


def _finished(path, rows):
    if not os.path.isfile(path):
        return False
    try:
        return np.load(path, mmap_mode="r").shape[0] == rows
    except ValueError:
        return False


@hydra.main(
    version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train"
)
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from openwam.dataloader.registry import build_dataset
    from openwam.model.video_backbone.wan import encode as wan_encode
    from turbo_physai.optimizations.models.openwam import (
        configure_backends,
        enabled,
        prepare_runtime_constants,
        text_store,
    )
    from turbo_physai.optimizations.models.openwam import latent_cache as store
    from openwam.train.openwam_trainer import OpenWAMTrainer

    configure_backends()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    root = store.directory()

    dataset = build_dataset(cfg.dataloader, split="train")
    readers = store.readers(dataset)
    for reader in readers:
        store.check_windows(reader)
    # The training constructor gives the VAE its training dtype, layout and compile settings.
    trainer = OpenWAMTrainer(cfg, accelerator=None, dataset=dataset)
    architecture = trainer.architecture
    store.validate_backbone(architecture)
    backbone = architecture.video_backbone
    signature = store.store_signature(dataset, store.vae_fingerprint(backbone.vae))
    device = torch.device("cuda", local_rank)
    architecture.set_dtype_device(architecture.dtype, device)
    backbone.vae.to(device=device)
    prepare_runtime_constants(architecture)

    meta_path = os.path.join(root, store.META)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            # A store built for all tasks can resume a selected task without
            # rewriting its metadata or invalidating the other cached readers.
            store.validate_signature(json.load(handle), signature)
    elif rank == 0:
        os.makedirs(root, exist_ok=True)
        with open(f"{meta_path}.tmp", "w", encoding="utf-8") as handle:
            json.dump(signature, handle)
        os.replace(f"{meta_path}.tmp", meta_path)

    episodes = [
        (r, e)
        for r, reader in enumerate(readers)
        for e in range(len(reader._episode_files))
        if store.episode_starts(reader, e)
    ]
    mine = [
        (r, e)
        for r, e in episodes[rank::world]
        if not _finished(
            store.episode_path(root, readers[r], e),
            len(store.episode_starts(readers[r], e)),
        )
    ]
    total = sum(len(store.episode_starts(readers[r], e)) for r, e in mine)
    logger.info(
        "[latent store] rank %d/%d: %d of %d episodes to encode, %d windows",
        rank,
        world,
        len(mine),
        len(episodes[rank::world]),
        total,
    )
    workers = min(
        int(cfg.training.dataset_num_workers), 4
    )  # Whole episodes are large in shared memory.
    loader = torch.utils.data.DataLoader(
        _Episodes(readers, mine),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        **({"prefetch_factor": 2} if workers else {}),
    )
    batch_size = int(cfg.training.batch_size)
    resident, rows, pending = {}, {}, []
    done, batches, started = 0, 0, time.monotonic()

    def encode(windows):
        nonlocal done, batches
        real = len(windows)
        # Pad the tail so the compiled encoder only ever sees one batch size.
        windows = windows + [windows[-1]] * (batch_size - real)
        pixels = torch.cat(
            [
                _window_pixels(resident[key], readers[key[0]], start, backbone.dtype)
                for key, start in windows
            ]
        )
        latents = wan_encode.encode_video(pixels, vae=backbone.vae, encoder=None)
        latents = latents.to(dtype=backbone.dtype).cpu()
        for (key, _), latent in zip(windows[:real], latents[:real]):
            rows[key].append(latent.contiguous())
            if len(rows[key]) == len(store.episode_starts(readers[key[0]], key[1])):
                _write(store.episode_path(root, readers[key[0]], key[1]), rows.pop(key))
                del resident[key]
        done += real
        batches += 1
        if rank == 0 and batches % 50 == 0:
            rate = done / (time.monotonic() - started)
            logger.info(
                "[latent store] rank 0: %d/%d windows, %.1f windows/s",
                done,
                total,
                rate,
            )

    with torch.no_grad():
        for r, e, frames in loader:
            key = (r, e)
            resident[key] = frames.pin_memory().to(device=device, non_blocking=True)
            rows[key] = []
            pending.extend(
                (key, start) for start in store.episode_starts(readers[r], e)
            )
            while len(pending) >= batch_size:
                encode(pending[:batch_size])
                pending = pending[batch_size:]
        if pending:
            encode(pending)
    if rows:
        raise RuntimeError(f"Unfinished episodes after encoding: {sorted(rows)}")
    logger.info(
        "[latent store] rank %d finished %d windows in %.0f s",
        rank,
        done,
        time.monotonic() - started,
    )
    if enabled("text_cache"):
        shard = text_store.generate(
            dataset,
            backbone,
            device=device,
            batch_size=batch_size,
            rank=rank,
            world=world,
        )
        text_store.finalize_distributed_index(shard)


if __name__ == "__main__":
    main()
