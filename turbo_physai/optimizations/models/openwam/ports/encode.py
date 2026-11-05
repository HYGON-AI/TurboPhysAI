# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from openwam.model.video_backbone.wan.encode import (
    _encode_text_uncached as _encode_text,
    encode_text_cached,
)
from turbo_physai.optimizations.models.openwam.settings import (
    enabled as _enabled,
    port_wrapper as _port_wrapper,
)

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
        ctx, sl = encode_text_cached(
            prompts,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            device=device,
            cache=cache,
        )
    else:
        ctx, sl = _encode_text(
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
