# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
OpenVLA 文本token定长 padding（分桶）。
为什么做：尽量规范以下shape，提高torch显存池的缓存命中率，不要频繁hipMalloc带来性能开销。
"""

from __future__ import annotations

import functools

import torch

BUCKET = 8


def _pad(batch, collator, ignore_index: int):
    """把 batch 长度向上取整到 ``BUCKET`` 的倍数。"""

    ids, labels = batch["input_ids"], batch["labels"]
    length = int(ids.size(1))
    target = min(
        ((length + BUCKET - 1) // BUCKET) * BUCKET,
        int(collator.model_max_length),
    )
    if target == length:
        return batch

    pad = target - length
    ids = torch.cat(
        [ids, ids.new_full((ids.size(0), pad), collator.pad_token_id)], dim=1
    )
    labels = torch.cat(
        [labels, labels.new_full((labels.size(0), pad), ignore_index)], dim=1
    )
    batch.update(input_ids=ids, labels=labels, attention_mask=ids.ne(collator.pad_token_id))
    return batch


def dataloader_iter_wrapper(original, options=None):
    """``wrap`` 工厂：``DataLoader`` 迭代时对 action collator 的 batch 分桶。"""

    del options

    from prismatic.util.data_utils import (
        IGNORE_INDEX,
        PaddedCollatorForActionPrediction,
    )

    @functools.wraps(original)
    def __iter__(self):
        collator = getattr(self, "collate_fn", None)
        if not isinstance(collator, PaddedCollatorForActionPrediction):
            yield from original(self)
            return
        for batch in original(self):
            yield _pad(batch, collator, IGNORE_INDEX)

    return __iter__
