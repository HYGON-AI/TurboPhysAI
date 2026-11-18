# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
flex attention 的 fp32 dot 改走 tf32。
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_PRECISION = "tf32"
_CACHE_TAG = "flex-attn-tf32"


def _tag_compile_cache(tag: str) -> None:
    """让 tf32 产物与旧的 ieee 产物分开，避免 FX graph cache 命中旧图。"""

    import torch.compiler.config as compiler_config

    current = str(getattr(compiler_config, "cache_key_tag", "") or "")
    if tag in current:
        return
    merged = f"{current}-{tag}" if current else tag
    compiler_config.cache_key_tag = merged
    logger.info("torch.compiler.config.cache_key_tag: %r -> %r", current, merged)


def wrap_get_float32_precision(original, _options=None):
    """
    Group 入口：返回恒为tf32的get_float32_precision。
    """

    _tag_compile_cache(_CACHE_TAG)

    @functools.wraps(original)
    def get_float32_precision():
        return f"'{_PRECISION}'"

    logger.info("flex attention fp32 dot: ieee -> tf32 (gfx936 v_mmac tf32)")
    return get_float32_precision
