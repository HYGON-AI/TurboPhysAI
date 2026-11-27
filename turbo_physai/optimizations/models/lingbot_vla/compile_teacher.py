# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import functools
import logging
import os
import types

import torch

logger = logging.getLogger(__name__)

_COMPILE_KWARGS = {"dynamic": True}


def _disabled() -> bool:
    """TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1 时跳过本组全部 torch.compile。"""

    return os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1"


def wrap_infer_feat(original, _options=None):
    if _disabled():
        return original

    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(type(self).forward_feat, **_COMPILE_KWARGS)
            logger.info("lingbot-vla: compiling MDM teacher forward_feat")
        proxy = copy.copy(self)
        proxy.forward_feat = types.MethodType(compiled, proxy)
        return original(proxy, *args, **kwargs)

    return wrapped
