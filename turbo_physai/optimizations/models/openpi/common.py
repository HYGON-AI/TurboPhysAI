# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Small helpers shared by independent Turbo replacement modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch


def option_bool(options: Mapping[str, Any], name: str, default: bool) -> bool:
    value = options.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def function_owner(function: Any) -> Any | None:
    """Find the model object captured by an OpenPI nested function."""

    for cell in getattr(function, "__closure__", ()) or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if value.__class__.__module__.startswith("openpi.models_pytorch"):
            return value
    return None


def compile_cached(
    function: Callable[..., Any],
    owner: Any | None,
    cache_key: str,
    options: Mapping[str, Any],
    *,
    key_function: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    if not option_bool(options, "compile", True):
        return function

    # Match the native OpenPI compile path: only the compile mode is passed to
    # torch.compile.  In particular, do not turn an explicit ``fullgraph:
    # false`` in the Turbo recipe into a different call signature; false is
    # already torch.compile's default.
    mode = options.get("mode", "default")
    kwargs: dict[str, Any] = {"mode": mode} if mode is not None else {}

    if owner is None:
        return torch.compile(function, **kwargs)

    # Use the same per-model cache slot and key shape as the reference patch.
    # The function code distinguishes the main and image regions, while mode
    # and direct/checkpoint topology distinguish independent compilations.
    cache = getattr(owner, "_compiled_checkpoint_regions", None)
    if cache is None:
        cache = {}
        setattr(owner, "_compiled_checkpoint_regions", cache)
    direct = option_bool(options, "direct", True)
    cache_code = getattr(key_function or function, "__code__", None)
    cached = cache.get((cache_code, mode, direct))
    if cached is None:
        import logging
        logging.getLogger(__name__).info("Turbo compiled checkpoint region cache_key=%s mode=%s direct=%s", cache_key, mode, direct)
        cached = torch.compile(function, **kwargs)
        cache[(cache_code, mode, direct)] = cached
    return cached


def run_checkpoint_region(
    checkpoint: Callable[..., Any],
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    cache_key: str,
    options: Mapping[str, Any],
) -> Any:
    """Match the patch's direct versus checkpointed compile topology."""

    owner = function_owner(function)
    if option_bool(options, "direct", True):
        compiled = compile_cached(function, owner, cache_key, options, key_function=function)
        return compiled(*args, **kwargs)

    def checkpoint_region(*region_args, **region_kwargs):
        checkpoint_kwargs = dict(region_kwargs)
        checkpoint_kwargs["use_reentrant"] = False
        checkpoint_kwargs["preserve_rng_state"] = False
        return checkpoint(function, *region_args, **checkpoint_kwargs)

    compiled = compile_cached(
        checkpoint_region,
        owner,
        cache_key,
        options,
        key_function=function,
    )
    return compiled(*args, **kwargs)
