# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
激活检查点策略：关掉前 N 个 action expert 层的整层重算。
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger("lingbotvla.turbophysai.ckpt_policy")

_EXPERT_LAYER_PATH = ("model", "qwenvl_with_expert", "qwen_expert", "model", "layers")
_EXPERT_LAYER_CLASS = "Qwen2DecoderLayer"


def _expert_layers(model):
    obj = model
    for name in _EXPERT_LAYER_PATH:
        obj = getattr(obj, name, None)
        if obj is None:
            return []
    return [m for m in obj if m.__class__.__name__ == _EXPERT_LAYER_CLASS]


def _int_option(options, key, default=0):
    try:
        return int((options or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def enable_gradient_checkpointing(original, _options=None):
    num_layers = _int_option(_options, "expert_layers")

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        layers = _expert_layers(self)[: max(0, num_layers)]
        for layer in layers:
            layer.gradient_checkpointing = False
        logger.info("[ckpt-policy] checkpointing disabled on %d expert layers", len(layers))
        return result

    return wrapped
