# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Offline frozen T5 contexts, generated alongside the RoboTwin VAE store.

Generation writes to text_cache_dir, or else to the ``text``
subdirectory of vae_latent_cache_dir. Training with
text_cache reads this store when text_cache_dir is set, or
when vae_latent_cache is enabled with its directory. A latent
directory left in the environment of a run without the latent cache does not
select it; the original bounded online text cache remains available.
Only valid token rows are saved, independently of TEXT_TRIM; readers restore
zero padding to the batch maximum or the full 512-token length. Version 1
stores must be removed and regenerated before using this format.
Generation validates entries and publishes a complete ``index.json``. Training
checks identity/index coverage and opens tensor files lazily, without sidecars.
"""

from turbo_physai.optimizations.models.openwam.settings import value

import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict

import numpy as np
import torch

from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.text import _native_t5, frozen_eval

logger = logging.getLogger(__name__)
FULL_LENGTH = 512
VERSION = 2
INDEX = "index.json"
DTYPES = {
    "bfloat16": (torch.bfloat16, np.dtype("int16")),
    "float16": (torch.float16, np.dtype("float16")),
    "float32": (torch.float32, np.dtype("float32")),
}


def directory():
    root = value("text_cache_dir", "").strip()
    if not root:
        latent_root = value("vae_latent_cache_dir", "").strip()
        root = os.path.join(latent_root, "text") if latent_root else None
    return root


def requested():
    """Whether training reads the offline store instead of encoding with T5."""
    if not enabled("text_cache"):
        return False
    if value("text_cache_dir", "").strip():
        return True
    return enabled("vae_latent_cache") and directory() is not None


def training_prompts(dataset):
    from turbo_physai.optimizations.models.openwam.ports.data import _training_prompts
    from turbo_physai.optimizations.models.openwam.latent_cache import (
        episode_starts,
        readers,
    )

    prompts = set()
    for reader in readers(dataset):
        for index, path in enumerate(reader._episode_files):
            if episode_starts(reader, index):
                prompts.update(
                    _training_prompts(reader._instructions, path, reader.task_name)
                )
    return sorted(prompts)


def validate_backbone(backbone):
    encoder = getattr(backbone, "text_encoder", None)
    if encoder is None or not frozen_eval(encoder) or not _native_t5(encoder):
        raise ValueError(
            "Offline text caching requires the native frozen Wan T5 in eval mode"
        )


def signature(backbone):
    """Hash exact weight bytes and tokenizer behavior, without FP64 weight copies."""
    validate_backbone(backbone)
    encoder = backbone.text_encoder
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(value.shape)}|{value.dtype}".encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    settings = (
        "dim",
        "dim_attn",
        "dim_ffn",
        "num_heads",
        "head_dim",
        "num_layers",
        "num_buckets",
        "shared_pos",
        "eps",
        "bidirectional",
        "max_dist",
    )
    digest.update(
        json.dumps(
            [
                (
                    name,
                    type(module).__name__,
                    {
                        key: getattr(module, key)
                        for key in settings
                        if hasattr(module, key)
                    },
                )
                for name, module in encoder.named_modules()
            ],
            sort_keys=True,
        ).encode()
    )
    tokenizer = backbone._tokenizer
    native = tokenizer.tokenizer
    backend = getattr(native, "backend_tokenizer", None)
    sentencepiece = getattr(native, "sp_model", None)
    if backend is not None:
        # HF calls mutate these transient backend fields. Our encoder supplies
        # fixed padding/truncation arguments, so they are not tokenizer identity.
        state = json.loads(backend.to_str())
        state.pop("padding", None)
        state.pop("truncation", None)
        tokenizer_state = json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
    elif sentencepiece is not None:
        tokenizer_state = sentencepiece.serialized_model_proto()
    else:
        raise ValueError(
            "Offline text caching requires a serializable HF/SentencePiece tokenizer"
        )
    token_digest = hashlib.sha256(tokenizer_state)
    token_digest.update(
        json.dumps(
            {
                "vocab": native.get_vocab(),
                "special_tokens": native.special_tokens_map,
                "added_tokens": native.get_added_vocab(),
                "clean": tokenizer.clean,
                "seq_len": tokenizer.seq_len,
                "padding_side": native.padding_side,
                "truncation_side": native.truncation_side,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    )
    dtype = str(next(encoder.parameters()).dtype).removeprefix("torch.")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported T5 cache dtype: {dtype}")
    return {
        "version": VERSION,
        "encoder": digest.hexdigest(),
        "tokenizer": token_digest.hexdigest(),
        "dtype": dtype,
        "length": FULL_LENGTH,
        "dim": int(encoder.dim),
    }


def _atomic(path, write):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temporary, "wb") as handle:
            write(handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, value):
    _atomic(
        path, lambda handle: handle.write(json.dumps(value, sort_keys=True).encode())
    )


def _key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def _path(root, prompt):
    key = _key(prompt)
    return os.path.join(root, key[:2], key + ".npy")


def _validate_version(root, meta):
    if meta.get("version") != VERSION:
        raise ValueError(
            f"Unsupported T5 text cache version: remove the text cache and rerun step 1 ({root})"
        )


def _entry(root, prompt, meta):
    _validate_version(root, meta)
    path = _path(root, prompt)
    with open(path + ".json", encoding="utf-8") as handle:
        record = json.load(handle)
    length = record["length"]
    if (
        record.get("prompt") != prompt
        or type(length) is not int
        or not 0 <= length <= FULL_LENGTH
    ):
        raise ValueError(f"Invalid T5 text cache entry: {path}")
    return length, _array(root, prompt, meta, length)


def _array(root, prompt, meta, length):
    path = _path(root, prompt)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (length, meta["dim"]) or array.dtype != DTYPES[meta["dtype"]][1]:
        raise ValueError(f"T5 text cache shape/dtype mismatch: {path}")
    return array


def _read_index(root, meta):
    path = os.path.join(root, INDEX)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"T5 text cache has no {INDEX}; rerun step 1 to build it ({root})"
        )
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    if (
        not isinstance(index, dict)
        or index.get("signature") != meta
        or index.get("complete") is not True
        or not isinstance(index.get("lengths"), dict)
    ):
        raise ValueError(
            f"T5 text cache index is incomplete or mismatched; rerun step 1 ({root})"
        )
    return index["lengths"]


def finalize_index(shards):
    """Publish one complete index from validated rank shards, without reopening entries."""
    if not shards:
        raise ValueError("Incomplete T5 text cache index ranks")
    first = shards[0]
    root, meta, world = first["root"], first["signature"], first["world"]
    if len(shards) != world or {shard["rank"] for shard in shards} != set(range(world)):
        raise ValueError("Incomplete T5 text cache index ranks")
    entries = {}
    for shard in shards:
        if any(
            shard[name] != first[name]
            for name in ("root", "signature", "world", "prompt_count", "prompt_digest")
        ):
            raise ValueError("T5 text cache index ranks disagree on settings/prompts")
        if entries.keys() & shard["lengths"].keys():
            raise ValueError("Overlapping T5 text cache index rank shards")
        entries.update(shard["lengths"])
    if len(entries) != first["prompt_count"]:
        raise ValueError("Incomplete T5 text cache index prompt coverage")
    # Preserve earlier task subsets when appending or resuming the same store.
    previous = {}
    if os.path.isfile(os.path.join(root, INDEX)):
        try:
            previous = _read_index(root, meta)
        except (OSError, ValueError, KeyError):
            logger.warning(
                "[text store] rebuilding invalid index from the current validated prompts: %s",
                root,
            )
    previous.update(entries)
    _json(
        os.path.join(root, INDEX),
        {"signature": meta, "complete": True, "lengths": previous},
    )


def finalize_distributed_index(shard):
    """Gather CPU metadata at the generator's end and publish only after every rank finishes."""
    if shard["world"] == 1:
        return  # generate() already published the single-rank index.
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", rank=shard["rank"], world_size=shard["world"]
    )
    try:
        shards = [None] * shard["world"] if shard["rank"] == 0 else None
        dist.gather_object(shard, shards, dst=0)
        error = [None]
        if shard["rank"] == 0:
            try:
                finalize_index(shards)
            except (OSError, ValueError, KeyError) as exc:
                error[0] = str(exc)
        # Inform every rank of publication failures instead of leaving peers waiting.
        dist.broadcast_object_list(error, src=0)
        if error[0] is not None:
            raise ValueError(f"T5 text cache index publication failed: {error[0]}")
    finally:
        dist.destroy_process_group()


def _finished(root, prompt, meta):
    try:
        _entry(root, prompt, meta)
        return True
    except (OSError, ValueError, KeyError, EOFError):
        return False


def generate(dataset, backbone, *, device, batch_size, rank=0, world=1):
    """Validate/encode each rank's unique prompts and return lengths for index publication."""
    from turbo_physai.optimizations.models.openwam.ports.encode import _encode_text

    root = directory()
    if root is None:
        raise ValueError("Set text_cache_dir or vae_latent_cache_dir")
    if batch_size < 1:
        raise ValueError("T5 cache batch size must be positive")
    if world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid T5 cache rank/world")
    meta = signature(backbone)
    meta_path = os.path.join(root, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            existing = json.load(handle).get("signature", {})
        _validate_version(root, existing)
        if existing != meta:
            raise ValueError(
                f"T5 text cache settings/weights differ: use a new directory ({root})"
            )
    else:
        _json(
            meta_path,
            {"signature": meta, "generation": {"t5_prefix": enabled("t5_prefix")}},
        )
    prompts = training_prompts(dataset)
    assigned, mine, entries = prompts[rank::world], [], {}
    started = time.monotonic()
    for index, prompt in enumerate(assigned, 1):
        try:
            entries[_key(prompt)] = _entry(root, prompt, meta)[0]
        except (OSError, ValueError, KeyError, EOFError):
            mine.append(prompt)
        if index % 4096 == 0:
            pass
    encoder = backbone.text_encoder
    if mine:
        encoder.to(device=device)
    with torch.no_grad():
        for start in range(0, len(mine), batch_size):
            batch = mine[start : start + batch_size]
            context, lengths = _encode_text(
                batch,
                tokenizer=backbone._tokenizer,
                text_encoder=encoder,
                device=device,
            )
            context, lengths = context.cpu().contiguous(), lengths.cpu().tolist()
            if (
                context.shape != (len(batch), FULL_LENGTH, meta["dim"])
                or context.dtype != DTYPES[meta["dtype"]][0]
            ):
                raise ValueError(
                    f"Unexpected T5 output shape/dtype: {context.shape}, {context.dtype}"
                )
            for prompt, value, length in zip(batch, context, lengths):
                if type(length) is not int or not 0 <= length <= FULL_LENGTH:
                    raise ValueError(f"Invalid T5 output length: {length}")
                path = _path(root, prompt)
                value = value[:length].contiguous()
                array = (
                    value.view(torch.int16).numpy()
                    if value.dtype == torch.bfloat16
                    else value.numpy()
                )
                _atomic(
                    path,
                    lambda handle, array=array: np.save(
                        handle, array, allow_pickle=False
                    ),
                )
                # Written last: an interrupted tensor write is never a complete entry.
                _json(path + ".json", {"prompt": prompt, "length": length})
                entries[_key(prompt)] = length
    shard = {
        "root": root,
        "signature": meta,
        "rank": rank,
        "world": world,
        "prompt_count": len(prompts),
        "prompt_digest": hashlib.sha256(
            "".join(_key(prompt) for prompt in prompts).encode()
        ).hexdigest(),
        "lengths": entries,
    }
    if world == 1:
        finalize_index([shard])
    return shard


class PromptContexts:
    """CPU memory maps; GPU holds only this batch's padded context, never a prompt LRU."""

    def __init__(self, root, meta, prompts):
        self.root, self.meta, self.lengths, self.maps = root, meta, {}, OrderedDict()
        indexed = _read_index(root, meta)
        for prompt in prompts:
            length = indexed.get(_key(prompt))
            if type(length) is not int or not 0 <= length <= FULL_LENGTH:
                raise ValueError(
                    f"T5 text cache index missing/invalid for {prompt!r}; rerun step 1 ({root})"
                )
            self.lengths[prompt] = length

    def __call__(self, prompts, device):
        missing = [prompt for prompt in prompts if prompt not in self.lengths]
        if missing:
            raise ValueError(
                f"Prompt absent from offline T5 cache: {missing[0]!r}; rerun step 1"
            )
        lengths = torch.tensor(
            [self.lengths[prompt] for prompt in prompts], dtype=torch.long
        )
        stop = max(1, int(lengths.max())) if enabled("text_trim") else FULL_LENGTH
        dtype = DTYPES[self.meta["dtype"]][0]
        target = torch.device(device)
        context = torch.zeros(
            (len(prompts), stop, self.meta["dim"]),
            dtype=dtype,
            pin_memory=target.type == "cuda",
        )
        for index, prompt in enumerate(prompts):
            array = self.maps.pop(prompt, None)
            if array is None:
                try:
                    array = _array(self.root, prompt, self.meta, self.lengths[prompt])
                except (OSError, ValueError, EOFError) as exc:
                    raise ValueError(
                        f"T5 text cache missing/incomplete for {prompt!r}; rerun step 1 ({self.root})"
                    ) from exc
            self.maps[prompt] = array
            while len(self.maps) > 64:
                self.maps.popitem(last=False)
            length = self.lengths[prompt]
            value = torch.from_numpy(np.array(array, copy=True))
            context[index, :length].copy_(
                value.view(torch.bfloat16) if dtype == torch.bfloat16 else value
            )
        # Lengths intentionally stay on CPU, as in the padding-plan path.
        return context.to(device=target, non_blocking=True), lengths


def offload(backbone):
    validate_backbone(backbone)
    backbone.text_encoder.to(device="cpu")
    # The older online LRU can otherwise retain GPU contexts after a mode switch.
    for name in ("_openwam_text_cache", "_openwam_text_cache_signature"):
        backbone.text_encoder.__dict__.pop(name, None)


def attach(dataset, backbone):
    """Check identity and indexed prompt coverage; tensor files are opened only on use."""
    root = directory()
    path = os.path.join(root, "meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"T5 text cache has no meta.json; run step 1 with text_cache=True ({root})"
        )
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)["signature"]
    _validate_version(root, meta)
    started = time.monotonic()
    if meta != signature(backbone):
        raise ValueError(f"T5 text cache weights/tokenizer/settings mismatch ({root})")
    prompts = training_prompts(dataset)
    contexts = PromptContexts(root, meta, prompts)
    offload(backbone)
    backbone.text_encoder._openwam_text_store = contexts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
