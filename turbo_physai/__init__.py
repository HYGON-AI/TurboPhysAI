# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""TurboPhysAI public API.

The optimization engine entry points are intentionally pure Python. Operator modules
are loaded lazily so planning and checking work without Torch or HCU installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .engine import errors as _errors
from .engine.definitions import group, replace, replace_import, wrap
from .engine.contracts import CompatibilityContext, CompatibilityResult
from . import optimizations as _optimizations  # noqa: F401 - registers catalogs



OptimizationExecutionError = _errors.OptimizationExecutionError
OptimizationRollbackError = _errors.OptimizationRollbackError
OptimizationConfigNotFoundError = _errors.OptimizationConfigNotFoundError
OptimizationConfigError = _errors.OptimizationConfigError
RuntimeConfigError = _errors.RuntimeConfigError


def apply(*args: Any, **kwargs: Any):
    from .engine import apply as _apply

    return _apply(*args, **kwargs)


def check(*args: Any, **kwargs: Any):
    from .engine import check as _check

    return _check(*args, **kwargs)


_LAZY_OPERATORS = {
    "MultiScaleDeformableAttnFunction": (
        "turbo_physai.operators.multi_scale_deformable_attn",
        "MultiScaleDeformableAttnFunction",
    ),
    "ModulatedDeformConv2dFunction": (
        "turbo_physai.operators.modulated_deform_conv",
        "ModulatedDeformConv2dFunction",
    ),
    "modulated_deform_conv2d": (
        "turbo_physai.operators.modulated_deform_conv",
        "modulated_deform_conv2d",
    ),
    "grid_sample": ("turbo_physai.operators.grid_sample", "grid_sample"),
    "interpolate": ("turbo_physai.operators.upsample_bilinear_2d", "interpolate"),
    "deformable_aggregation_function": (
        "turbo_physai.operators.deformable_aggregation",
        "deformable_aggregation_function",
    ),
    "DeformableAggregationFunction": (
        "turbo_physai.operators.deformable_aggregation",
        "DeformableAggregationFunction",
    ),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_OPERATORS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "apply",
    "check",
    "OptimizationExecutionError",
    "OptimizationRollbackError",
    "OptimizationConfigNotFoundError",
    "OptimizationConfigError",
    "RuntimeConfigError",
    "group",
    "replace",
    "replace_import",
    "wrap",
    "CompatibilityContext",
    "CompatibilityResult",
] + sorted(_LAZY_OPERATORS)
