# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""把 embed_tokens 从 FSDP 根单元拆出来单独 fully_shard。

根单元（visual + embed_tokens，约 956M 参数）backward 末尾要一次性分配 3.65 GiB fp32 梯度缓冲；
拆成独立 unit 后最大单个缓冲降到 2.6 GiB。
"""

from __future__ import annotations

import functools


def _find_embed_tokens(module):
    lm = getattr(getattr(getattr(module, "model", None), "qwenvl_with_expert", None), "qwenvl", None)
    lm = getattr(lm, "model", None)
    if lm is None:
        return None
    for owner in (getattr(lm, "language_model", None), lm):
        embed = getattr(owner, "embed_tokens", None)
        if embed is not None:
            return embed
    return None


def fully_shard(original, _options=None):
    @functools.wraps(original)
    def wrapped(module, *args, **kwargs):
        embed = _find_embed_tokens(module)
        if embed is not None and not embed.__dict__.get("_tp_split_sharded"):
            embed.__dict__["_tp_split_sharded"] = True
            original(embed, *args, **kwargs)
        return original(module, *args, **kwargs)

    return wrapped
