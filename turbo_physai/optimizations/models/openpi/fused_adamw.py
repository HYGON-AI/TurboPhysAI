# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent fused AdamW replacement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def _is_gpu_float_param(param: Any) -> bool:
    """Accept tensors and the named-parameter pairs supported by PyTorch."""
    if isinstance(param, tuple) and len(param) == 2 and isinstance(param[0], str):
        param = param[1]
    return isinstance(param, torch.Tensor) and param.device.type == "cuda" and torch.is_floating_point(param)


def _prepare_params(params: Any) -> tuple[Any, bool]:
    """Materialize iterators once so both the eligibility check and AdamW see them."""
    if isinstance(params, torch.Tensor):
        return params, False
    try:
        params_iterator = iter(params)
    except TypeError:
        return params, False
    # Do not swallow errors raised while iterating a user's generator.
    params = list(params_iterator)

    if not params:
        return params, False

    if not isinstance(params[0], dict):
        return params, all(_is_gpu_float_param(param) for param in params)

    # Respect per-group choices as well as constructor keyword arguments.
    if any(
        not isinstance(group, dict)
        or "params" not in group
        or "fused" in group
        or "foreach" in group
        or group.get("differentiable", False)
        for group in params
    ):
        return params, False

    eligible = True
    has_params = False
    for group in params:
        group_params = group["params"]
        if isinstance(group_params, torch.Tensor):
            group_params = [group_params]
        elif isinstance(group_params, set):
            # Let PyTorch report its own unsupported unordered-collection error.
            eligible = False
            continue
        else:
            try:
                group_iterator = iter(group_params)
            except TypeError:
                eligible = False
                continue
            group_params = list(group_iterator)
            # Optimizer.add_param_group also materializes and updates this dict.
            group["params"] = group_params
        has_params = has_params or bool(group_params)
        eligible = eligible and all(_is_gpu_float_param(param) for param in group_params)
    return params, eligible and has_params


def wrap_fused_adamw(original, options: Mapping[str, Any]):
    fused_enabled = bool(options.get("fused", True))

    def wrapped(*args, **kwargs):
        if (
            not fused_enabled
            or "fused" in kwargs
            or "foreach" in kwargs
            or kwargs.get("differentiable", False)
            or (args and "params" in kwargs)
            or (not args and "params" not in kwargs)
        ):
            return original(*args, **kwargs)

        params, eligible = _prepare_params(args[0] if args else kwargs["params"])
        if args:
            args = (params, *args[1:])
        else:
            kwargs = {**kwargs, "params": params}
        if eligible:
            kwargs = {**kwargs, "fused": True}
        return original(*args, **kwargs)

    return wrapped
