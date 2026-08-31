# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Static checks for direct Python object references."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..execution.replacements.base import HandlerError, resolve_replacement


def _dotted_name(node: ast.AST, aliases: Mapping[str, str]) -> Optional[str]:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value, aliases)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _aliases(
    tree: ast.Module,
    start_line: int,
    end_line: int,
) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    assignments = []
    top_level = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if id(node) not in top_level and not start_line <= line <= end_line:
            continue
        if isinstance(node, ast.Import):
            for item in node.names:
                bound = item.asname or item.name.split(".", 1)[0]
                aliases[bound] = item.name if item.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name != "*":
                    aliases[item.asname or item.name] = (
                        f"{node.module}.{item.name}"
                    )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            assignments.append(node)

    # Resolve simple aliases such as ``fast_op = public_impl.optimized_op``.
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            value = _dotted_name(node.value, aliases)
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else (node.target,)
            )
            if value is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != value:
                    aliases[target.id] = value
                    changed = True
        if not changed:
            break
    return aliases


def direct_reference_lines(
    replacement_path: str,
    candidates: Iterable[str],
) -> Tuple[Tuple[str, int], ...]:
    """Return direct candidate references found in one Replacement body."""

    try:
        replacement = resolve_replacement(replacement_path)
        source_file = inspect.getsourcefile(replacement)
        source_lines, start_line = inspect.getsourcelines(replacement)
    except (HandlerError, OSError, TypeError):
        return ()
    if source_file is None:
        return ()
    try:
        tree = ast.parse(Path(source_file).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return ()

    end_line = start_line + len(source_lines) - 1
    aliases = _aliases(tree, start_line, end_line)
    accepted = set(candidates)
    found = {}
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if not start_line <= line <= end_line:
            continue
        path = _dotted_name(node, aliases)
        if path in accepted:
            found.setdefault(path, line)
        if isinstance(node, ast.Call):
            loader = _dotted_name(node.func, aliases)
            if loader not in {"__import__", "importlib.import_module"}:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            module_name = node.args[0].value
            if not isinstance(module_name, str):
                continue
            for candidate in accepted:
                if candidate == module_name or candidate.startswith(
                    module_name + "."
                ):
                    found.setdefault(candidate, line)
    return tuple(sorted(found.items()))
