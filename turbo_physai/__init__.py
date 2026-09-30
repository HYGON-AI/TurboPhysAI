# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""TurboPhysAI public API.

The optimization engine entry points are intentionally pure Python. Operator modules
are loaded lazily so configuration tooling works without Torch or HCU installed.
Runtime application loads the dependencies required by the selected optimizations.
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


def __getattr__(name: str) -> Any:
    if name == "operators":
        value = import_module("turbo_physai.operators")
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "apply",
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
    "operators",
]
