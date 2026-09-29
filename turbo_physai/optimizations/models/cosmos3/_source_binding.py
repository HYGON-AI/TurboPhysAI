# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Bind unresolved global names of ported functions to their source module.

Ported Cosmos function bodies reference module-level names (helper functions,
imported classes, loggers).  Instead of hand-enumerating those imports for every
ported symbol, each port module calls :func:`bind_missing_globals` after its
definitions: any name still missing from the module globals is copied from the
original Cosmos module, so the ported code sees exactly the objects the source
file saw.  Names already defined in the port module (Turbo replacements such as
fused ops or ordered-comm helpers) are never overwritten.
"""

from __future__ import annotations

import builtins
import types
from typing import Any, Iterable


def _iter_code_names(code: types.CodeType) -> Iterable[str]:
    yield from code.co_names
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _iter_code_names(const)


def _iter_function_names(
    function: types.FunctionType,
    seen: set[int] | None = None,
) -> Iterable[str]:
    """Yield globals used by a function and wrapped functions in its closure."""

    if seen is None:
        seen = set()
    identity = id(function)
    if identity in seen:
        return
    seen.add(identity)

    yield from _iter_code_names(function.__code__)

    wrapped = getattr(function, "__wrapped__", None)
    if isinstance(wrapped, types.FunctionType):
        yield from _iter_function_names(wrapped, seen)

    for cell in function.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if isinstance(value, types.FunctionType):
            yield from _iter_function_names(value, seen)


def bind_missing_globals(module_globals: dict[str, Any], *source_modules: Any) -> None:
    """Fill missing global names referenced by functions in *module_globals*."""

    functions = [
        value
        for value in list(module_globals.values())
        if isinstance(value, types.FunctionType)
    ]
    needed: set[str] = set()
    for function in functions:
        needed.update(_iter_function_names(function))
    for name in sorted(needed):
        if name in module_globals or hasattr(builtins, name):
            continue
        for source in source_modules:
            if hasattr(source, name):
                module_globals[name] = getattr(source, name)
                break
