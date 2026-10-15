# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import types
import torch
import torch.nn as nn
import openwam.model.video_backbone.wan.encode as _source
from turbo_physai.optimizations.common.python.source_binding import bind_missing_globals
from turbo_physai.optimizations.models.openwam.host_async import (
    active as host_async_active,
    upload,
)
from turbo_physai.optimizations.models.openwam.text import cached_text, t5_prefix_plan
from turbo_physai.optimizations.models.openwam import enabled


def encode_text(
    prompts: list, *, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    return cached_text(
        prompts,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        encode=lambda batch: _encode_text(
            batch, tokenizer=tokenizer, text_encoder=text_encoder, device=device
        ),
    )


def _encode_text(
    prompts: list, *, tokenizer, text_encoder, device
) -> Tuple[Tensor, Tensor]:
    ids, mask = tokenizer(
        prompts,
        return_mask=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    full_length = ids.shape[1]
    # Keep native tokenizer lengths on the host for trim/packing. Consumers
    # needing device lengths already move them explicitly. No GPU readback.
    cpu_plan = enabled("padding_plan_cpu") and mask.device.type == "cpu"
    host_lengths = mask.gt(0).sum(dim=1).long() if cpu_plan else None
    prefix = t5_prefix_plan(ids, mask, text_encoder)
    if prefix is not None:
        stop, cpu_lengths = prefix
        ids = ids[:, :stop].contiguous()
        mask = mask[:, :stop].contiguous()
    if host_async_active():
        # A blocking pageable copy waits for every queued kernel, including the
        # optimizer step this encode is meant to overlap.
        ids = upload(ids, device)
        mask = upload(mask, device)
    else:
        ids = ids.to(device)
        mask = mask.to(device)
    seq_lens = host_lengths if cpu_plan else mask.gt(0).sum(dim=1).long()
    context = text_encoder(ids, mask)
    lengths_to_zero = (
        host_lengths.tolist()
        if cpu_plan
        else (seq_lens if prefix is None else cpu_lengths)
    )
    for i, v in enumerate(lengths_to_zero):
        context[i, v:] = 0
    if prefix is not None:
        context = torch.nn.functional.pad(
            context, (0, 0, 0, full_length - context.shape[1])
        )
    return context, seq_lens


bind_missing_globals(globals(), _source)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

encode_text_wrapper = _port_wrapper(
    encode_text, "openwam.model.video_backbone.wan.encode.encode_text", "openwam.text"
)
