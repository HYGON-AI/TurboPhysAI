# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile PI0 main/image regions through one official checkpoint-method wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.utils.checkpoint


_REGION_CACHE_KEYS = {
    "forward_func": "pi0_main_forward",
    "image_embed_func": "pi0_image_embed",
}


def _option_bool(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _function_owner(function: Any) -> Any | None:
    """Find the model captured by the official OpenPI nested region function."""
    for cell in getattr(function, "__closure__", ()) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if value.__class__.__module__.startswith("openpi.models_pytorch"):
            return value
    return None


def _compile_cached(
    function: Callable[..., Any],
    owner: Any | None,
    cache_key: str,
    options: Mapping[str, Any],
    *,
    key_function: Callable[..., Any],
) -> Callable[..., Any]:
    if not _option_bool(options, "compile", True):
        return function

    # Preserve the existing torch.compile call: only mode is passed, not an
    # explicit fullgraph=False or any new compilation option.
    mode = options.get("mode", "default")
    kwargs: dict[str, Any] = {"mode": mode} if mode is not None else {}
    if owner is None:
        return torch.compile(function, **kwargs)

    cache = getattr(owner, "_compiled_checkpoint_regions", None)
    if cache is None:
        cache = {}
        setattr(owner, "_compiled_checkpoint_regions", cache)
    direct = _option_bool(options, "direct", True)
    cache_code = getattr(key_function, "__code__", None)
    cached = cache.get((cache_code, mode, direct))
    if cached is None:
        import logging

        logging.getLogger(__name__).info(
            "Turbo compiled checkpoint region cache_key=%s mode=%s direct=%s",
            cache_key, mode, direct,
        )
        cached = torch.compile(function, **kwargs)
        cache[(cache_code, mode, direct)] = cached
    return cached


def _run_region(function, args, kwargs, *, cache_key, options):
    owner = _function_owner(function)
    if _option_bool(options, "direct", True):
        compiled = _compile_cached(function, owner, cache_key, options, key_function=function)
        return compiled(*args, **kwargs)

    checkpoint = torch.utils.checkpoint.checkpoint

    def checkpoint_region(*region_args, **region_kwargs):
        checkpoint_kwargs = dict(region_kwargs)
        checkpoint_kwargs["use_reentrant"] = False
        checkpoint_kwargs["preserve_rng_state"] = False
        return checkpoint(function, *region_args, **checkpoint_kwargs)

    compiled = _compile_cached(
        checkpoint_region, owner, cache_key, options, key_function=function,
    )
    return compiled(*args, **kwargs)


def wrap_pi0_compile(original, options: Mapping[str, Any]):
    """Compile both selected regions; keep the original path for every other case."""
    def wrapped(self, func, *args, **kwargs):
        cache_key = _REGION_CACHE_KEYS.get(getattr(func, "__name__", ""))
        if (
            cache_key is not None
            and getattr(self, "gradient_checkpointing_enabled", False)
            and self.training
        ):
            return _run_region(func, args, kwargs, cache_key=cache_key, options=options)
        return original(self, func, *args, **kwargs)

    return wrapped
