# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import torch
from typing import TYPE_CHECKING

from openwam.model.video_backbone.wan.encode import (
    TrainTextEmbedCache,
)
from turbo_physai.optimizations.models.openwam.settings import (
    enabled as _enabled,
    port_wrapper as _port_wrapper,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from openwam.model.video_backbone.wan.encode import Tensor, Tuple


def encode_text(
    prompts: list, *, tokenizer, text_encoder, device, cache=None
) -> Tuple[Tensor, Tensor]:
    store = getattr(text_encoder, "_openwam_text_store", None)
    if store is not None:
        ctx, sl = store(prompts, device)
        if cache is not None:
            cache.last_max_len = int(sl.max()) if len(sl) else 0
        return ctx, sl
    from turbo_physai.optimizations.models.openwam.text_store import requested

    if requested():
        raise RuntimeError(
            "Offline T5 cache requested but not attached; initialize it before encoding training text"
        )

    if cache is not None:
        ctx, sl = prepare_text_cached(
            prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            cache=cache,
        )()
    else:
        ctx, sl = _encode_text_uncached(
            prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
        )
    if _enabled("padding_plan_cpu") and sl.device.type != "cpu":
        sl = sl.cpu()
    return ctx, sl


encode_text_wrapper = _port_wrapper(
    encode_text, "openwam.model.video_backbone.wan.encode.encode_text", "openwam.text"
)


def _encode_text_uncached(
    prompts: list, *, tokenizer, text_encoder, device, cache=None
) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    cpu_plan = _enabled("padding_plan_cpu") and mask.device.type == "cpu"
    host_lengths = mask.gt(0).sum(dim=1).long() if cpu_plan else None
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = host_lengths if cpu_plan else mask.gt(0).sum(dim=1).long()
    context = _run_encoder(text_encoder, ids, mask, device=device, cache=cache)
    lengths_to_zero = host_lengths.tolist() if cpu_plan else seq_lens
    for i, v in enumerate(lengths_to_zero):
        context[i, v:] = 0
    return context, seq_lens


def _encoder_is_frozen_eval(text_encoder) -> bool:
    if not isinstance(text_encoder, torch.nn.Module):
        return False
    if any(m.training for m in text_encoder.modules()):
        return False
    return not any(p.requires_grad for p in text_encoder.parameters())


def _run_encoder(text_encoder, ids, mask, *, device, cache=None):
    offload = getattr(text_encoder, "_openwam_text_offload", None)
    if offload is None:
        offload = getattr(cache, "offload", None)
    if offload is None:
        return text_encoder(ids, mask)
    if isinstance(cache, TrainTextEmbedCache):
        offload.cache = cache
    if not _encoder_is_frozen_eval(text_encoder):
        offload.disable(device)
        return text_encoder(ids, mask)
    if not offload.enabled:
        return text_encoder(ids, mask)
    offload.acquire(device)
    try:
        return text_encoder(ids, mask)
    finally:
        offload.release()


def _pin_for_device(t: Tensor, device) -> Tensor:
    """Stage host data for a later non-blocking copy."""
    return t.pin_memory() if torch.device(device).type == "cuda" else t


def prepare_text_cached(prompts: list, *, tokenizer, text_encoder, device, cache):
    """Stage host inputs now and return a callable that queues the text work.

    The VAE-first path pins each batch's inputs before queuing video work.
    PyTorch tracks the non-blocking copies before recycling pinned storage.
    """
    if cache is None or not _encoder_is_frozen_eval(text_encoder):
        if cache is not None:
            cache.clear()
        return lambda: _encode_text_uncached(
            prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            cache=cache,
        )
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    cpu_plan = _enabled("padding_plan_cpu") and mask.device.type == "cpu"
    keys = [
        (ids[i].numpy().tobytes(), mask[i].numpy().tobytes())
        for i in range(ids.shape[0])
    ]
    seq_lens_cpu = mask.gt(0).sum(dim=1).long()
    cache.last_max_len = int(seq_lens_cpu.max().item()) if seq_lens_cpu.numel() else 0
    seq_lens_host = seq_lens_cpu if cpu_plan else _pin_for_device(seq_lens_cpu, device)
    rows = [cache.get(k) for k in keys]
    n = cache.hits + cache.misses + 1
    if n & (n - 1) == 0 and os.environ.get("RANK", "0") == "0":
        logger.info(
            "text-embed cache: call %d, hits %d, misses %d, entries %d",
            n,
            cache.hits,
            cache.misses,
            len(cache.rows),
        )
    hit = all(row is not None for row in rows)
    miss = None
    if not hit:
        full_len = ids.shape[1]
        if os.environ.get("OPENWAM_TEXT_MISS_FAST", "0") == "1":
            # Only encode distinct misses on a prefix rounded up to 64 tokens.
            miss_by_key = {}
            for i, (key, row) in enumerate(zip(keys, rows)):
                if row is None:
                    miss_by_key.setdefault(key, i)
            miss = list(miss_by_key.values())
            lengths = seq_lens_cpu[miss]
            width = min(full_len, max(64, ((int(lengths.max()) + 63) // 64) * 64))
            ids_host = _pin_for_device(ids[miss, :width].contiguous(), device)
            mask_host = _pin_for_device(mask[miss, :width].contiguous(), device)
            lengths_host = _pin_for_device(lengths, device)
        else:
            ids_host = _pin_for_device(ids, device)
            mask_host = _pin_for_device(mask, device)
            lengths_host = (
                _pin_for_device(seq_lens_cpu, device) if cpu_plan else seq_lens_host
            )

    def encode():
        seq_lens = (
            seq_lens_host if cpu_plan else seq_lens_host.to(device, non_blocking=True)
        )
        if hit:
            cache.hits += 1
            return torch.stack(rows, dim=0), seq_lens
        cache.misses += 1
        context = _run_encoder(
            text_encoder,
            ids_host.to(device, non_blocking=True),
            mask_host.to(device, non_blocking=True),
            device=device,
            cache=cache,
        )
        if miss is None and not cpu_plan:
            lengths_device = seq_lens
        else:
            lengths_device = lengths_host.to(context.device, non_blocking=True)
        pad = torch.arange(context.shape[1], device=context.device).unsqueeze(
            0
        ) >= lengths_device.unsqueeze(1)
        context.masked_fill_(pad.unsqueeze(-1), 0)
        if miss is None:
            for i, key in enumerate(keys):
                if cache.get(key) is None:
                    cache.put(key, context[i].clone())
            return context, seq_lens

        if context.shape[1] < full_len:
            context = torch.nn.functional.pad(
                context, (0, 0, 0, full_len - context.shape[1])
            )
        new_rows = {}
        for j, i in enumerate(miss):
            row = context[j].clone()
            new_rows[keys[i]] = row
            cache.put(keys[i], row)
        return torch.stack(
            [row if row is not None else new_rows[key] for row, key in zip(rows, keys)]
        ), seq_lens

    return encode
