# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


def _property_source(value: property) -> Optional[str]:
    """Return deterministic source for every accessor exposed by a property."""

    parts = []
    for name in ("fget", "fset", "fdel"):
        accessor = getattr(value, name)
        if accessor is None:
            continue
        try:
            source = inspect.getsource(accessor)
        except (OSError, TypeError):
            return None
        # The marker makes the accessor role part of both source and AST
        # evidence while keeping one evidence value per declared target.
        parts.append(
            f"__turbo_physai_property_accessor__ = {name!r}\n"
            f"{textwrap.dedent(source)}"
        )
    return "\n".join(parts) if parts else None


def _source(value: Any) -> Optional[str]:
    if isinstance(value, property):
        return _property_source(value)
    if isinstance(value, ModuleType):
        module_file = getattr(value, "__file__", None)
        if module_file:
            path = Path(module_file)
            if path.suffix.lower() in {".py", ".pyw"}:
                try:
                    return path.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    return None
    try:
        return inspect.getsource(value)
    except (OSError, TypeError):
        return None


def source_hash(value: Any) -> Optional[str]:
    source = _source(value)
    if source is not None:
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return "source-v1:" + digest

    # Native/PyBind callables have no Python source. Bind them to the loaded
    # extension artifact so generated Configs can still verify their identity.
    module = inspect.getmodule(value)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return None
    path = Path(module_file)
    if path.suffix.lower() not in {".so", ".dylib", ".dll", ".pyd"}:
        return None
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    return "artifact-v1:" + digest


def _canonical_ast(value: Any) -> str:
    if isinstance(value, ast.AST):
        normalized_fields = []
        for name, item in ast.iter_fields(value):
            if item is None or item == []:
                continue
            normalized_fields.append(f"{name}={_canonical_ast(item)}")
        return f"{type(value).__name__}({','.join(normalized_fields)})"
    if isinstance(value, list):
        return "[" + ",".join(_canonical_ast(item) for item in value) + "]"
    return repr(value)


def ast_hash(value: Any) -> Optional[str]:
    source = _source(value)
    if source is None:
        return None
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    normalized = _canonical_ast(tree)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return "ast-v1:" + digest
