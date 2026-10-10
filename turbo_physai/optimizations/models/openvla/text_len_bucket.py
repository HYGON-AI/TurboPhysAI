# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
OpenVLA fixed-length (bucketed) text padding wrapper for TurboPhysAI.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import Any, Optional

import torch

logger = logging.getLogger("openvla.data.text_len_bucket")


def _bucket_size(options: Optional[Mapping[str, Any]]) -> int:
    options = dict(options or {})
    raw = options.get("bucket", 8)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"openvla.data.text_len_bucket: options.bucket must be an integer, got {raw!r}"
        ) from exc


def _ignore_index() -> int:
    from prismatic.util.data_utils import IGNORE_INDEX
    return IGNORE_INDEX


def bucketed_collate_wrapper(original: Any, options: Optional[Mapping[str, Any]] = None):
    """Wrapper factory ``(original, options) -> fixed-length-padding ``__call__``.

    ``original`` is ``PaddedCollatorForActionPrediction.__call__``.  The returned
    function calls it unchanged and then rounds the batch up to a multiple of
    ``options.bucket`` tokens.
    """
    # Defensive: the framework only ever passes the resolved method here.  If
    # something else was resolved, stay a no-op rather than breaking collation.
    if not callable(original):
        return original

    bucket = _bucket_size(options)
    if bucket < 2:
        logger.info(
            "Fixed-length text padding disabled (options.bucket=%s < 2); baseline collation kept.",
            bucket,
        )
        return original

    # Fail fast (at apply time, before the first step) if the baseline constant
    # cannot be resolved -- better than padding with a wrong ignore index.
    ignore_index = _ignore_index()

    @functools.wraps(original)
    def __call__(self: Any, instances: Any) -> Any:
        batch = original(self, instances)

        input_ids = batch["input_ids"]
        labels = batch["labels"]
        batch_len = input_ids.size(1)

        # Round *up* to the next multiple, capped at `model_max_length` (which the
        # baseline already truncated `input_ids` to, so the cap only ever bites
        # within `bucket - 1` tokens of the limit).
        bucketed_len = min(
            ((batch_len + bucket - 1) // bucket) * bucket,
            int(self.model_max_length),
        )
        if bucketed_len <= batch_len:
            return batch

        pad_len = bucketed_len - batch_len
        input_ids = torch.cat(
            [input_ids, input_ids.new_full((input_ids.size(0), pad_len), fill_value=self.pad_token_id)],
            dim=1,
        )
        labels = torch.cat(
            [labels, labels.new_full((labels.size(0), pad_len), fill_value=ignore_index)],
            dim=1,
        )

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        # Same expression as the baseline computes before this point; the pads
        # added above are `pad_token_id` => `attention_mask = False`.
        batch["attention_mask"] = input_ids.ne(self.pad_token_id)
        return batch

    __call__.__doc__ = (
        f"{original.__doc__ or ''}\n\n"
        f"[openvla.data.text_len_bucket] Batch length is rounded up to a multiple of "
        f"{bucket} tokens (capped at `model_max_length`); extra positions are right-padding "
        f"with `pad_token_id` / label {ignore_index}."
    )

    logger.info(
        "Fixed-length text padding enabled via `openvla.data.text_len_bucket` "
        "(bucket=%d tokens, capped at model_max_length).",
        bucket,
    )
    return __call__
