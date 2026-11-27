# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import logging
import os

import torch

logger = logging.getLogger(__name__)

_COMPILE_KWARGS = {"dynamic": True}

_MODE = "default"

_BLOCK_MARK = "_tp_per_layer_compiled"
_BOUNDARY_MARK = "_tp_dynamo_boundary"


def _disabled() -> bool:
    """TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1 时跳过本组全部 torch.compile。"""

    return os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1"


def _resolve(root, path):
    node = root
    for name in path:
        node = getattr(node, name, None)
        if node is None:
            return None
    return node


def _install_vision(wrapper) -> None:
    if _disabled():
        return

    visual = _resolve(wrapper, ("qwenvl", "visual"))
    if visual is None:
        logger.warning("lingbot-vla: visual tower not found, skip per-layer compile")
        return

    compiled = 0
    for block in getattr(visual, "blocks", ()):
        if block.__dict__.get(_BLOCK_MARK, False):
            continue
        block.forward = torch.compile(block.forward, **_COMPILE_KWARGS)
        block.__dict__[_BLOCK_MARK] = True
        compiled += 1
    logger.info("lingbot-vla: per-layer compile on %d vision blocks", compiled)

    if not visual.__dict__.get(_BOUNDARY_MARK, False):
        visual.forward = torch._dynamo.disable(visual.forward)
        visual.__dict__[_BOUNDARY_MARK] = True
        logger.info("lingbot-vla: vision tower marked as dynamo boundary")


def wrap_torch_compile(original, _options=None):
    @functools.wraps(original)
    def patched_compile(model, *args, **kwargs):
        if _disabled():
            return model

        wrapper = getattr(getattr(model, "model", None), "qwenvl_with_expert", None)
        if wrapper is not None:
            _install_vision(wrapper)
            kwargs.setdefault("mode", _MODE)
        return original(model, *args, **kwargs)

    return patched_compile
