# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools

import torch

_CACHE = {}


def wrap_interpolate_pos_encoding_without_cls(original, options=None):
    @functools.wraps(original)
    def wrapped(self, x, h, w, input_pos_embed):
        key = (h, w, x.shape[1], x.dtype, input_pos_embed.data_ptr())
        out = _CACHE.get(key)
        if out is None:
            out = original(self, x, h, w, input_pos_embed)
            _CACHE[key] = out
        return out

    return torch._dynamo.disable(wrapped)
