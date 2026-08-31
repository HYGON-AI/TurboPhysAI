# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Small developer-facing declarations for optimization catalogs.

The execution framework continues to use :class:`ReplacementSpec` internally.  Catalog
authors normally describe only the group boundary and each target/replacement
pair through :func:`group`, :func:`replace`, :func:`replace_import`, and
:func:`wrap`.

Import-time compatibility declarations are intentionally kept in
``turbo_physai.compatibility`` so they do not expand the ordinary optimization
development interface.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

from ..checking.conflicts import find_target_conflicts, format_conflict
from ..contracts import Mechanism, OptimizationGroup, ReplacementSpec
from .registry import Registry, default_registry


@dataclass(frozen=True)
class Replacement:
    target: str
    replacement: str
    mechanism: Mechanism = Mechanism.REPLACE
    aliases: Tuple[str, ...] = ()
    runtime_condition: Optional[str] = None
    mechanism_options: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.target or not self.replacement:
            raise ValueError("target and replacement are required")
        if self.runtime_condition is not None and not self.runtime_condition:
            raise ValueError("runtime_condition must be a non-empty object path")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(
            self, "mechanism_options", dict(self.mechanism_options or {})
        )


@dataclass(frozen=True)
class Optimization:
    """Developer declaration plus the internal objects generated from it."""

    group_id: str
    replacements: Tuple[Replacement, ...]
    specs: Tuple[ReplacementSpec, ...]
    definition: OptimizationGroup

    def register(self, registry: Registry = default_registry) -> None:
        conflicts = find_target_conflicts(((self.group_id, self.specs),))
        if conflicts:
            raise ValueError(
                f"invalid optimization Group: {format_conflict(conflicts[0])}"
            )
        # Validate the complete declaration before mutating the Registry.  A
        # conflicting Group must not leave newly inserted orphan specs behind.
        for spec in self.specs:
            current = registry.get_spec(spec.replacement_id)
            if current is not None and current != spec:
                raise ValueError(f"duplicate ReplacementSpec id: {spec.replacement_id}")
        current_group = registry.get_group(self.definition.group_id)
        if current_group is not None and current_group != self.definition:
            raise ValueError(
                f"duplicate OptimizationGroup id: {self.definition.group_id}"
            )

        for spec in self.specs:
            if registry.get_spec(spec.replacement_id) is None:
                registry.register_spec(spec)
        if current_group is None:
            registry.register_group(self.definition)


def replace(
    target: str,
    replacement: str,
    *,
    aliases: Iterable[str] = (),
    runtime_condition: Optional[str] = None,
) -> Replacement:
    """Declare a direct replacement; function/class type is inferred on check."""

    return Replacement(
        target,
        replacement,
        Mechanism.REPLACE,
        tuple(aliases),
        runtime_condition,
    )


def replace_import(target: str, replacement: str) -> Replacement:
    """Declare complete module replacement before the target is imported.

    This restricted declaration is intended for model-private extension modules
    that cannot be imported in the target environment.  Both arguments are
    module paths; partial exports and conditional runtime dispatch are not
    supported.
    """

    return Replacement(target, replacement, Mechanism.IMPORT_REPLACE)


def import_alias(module: str, source: str, alias: str) -> Replacement:
    """Expose ``source`` from one module under an additional compatibility name."""

    if not module or not source or not alias or "." in source or "." in alias:
        raise ValueError("module, source and alias must be non-empty simple names")
    return Replacement(
        f"{module}.{alias}",
        f"{module}.{source}",
        Mechanism.IMPORT_ALIAS,
    )


def optional_import(module: str) -> Replacement:
    """Skip one exact optional module during subsequent model imports."""

    if not module or "." not in module:
        raise ValueError("optional module must be a complete module path")
    return Replacement(module, module, Mechanism.OPTIONAL_IMPORT)


def registry_override(
    module: str,
    registry: str,
    *,
    names: Iterable[str],
) -> Replacement:
    """Import a module while allowing only named Registry entries to override."""

    normalized = tuple(names)
    if (
        not module
        or not registry
        or not normalized
        or any(not isinstance(name, str) or not name for name in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("module, registry and unique non-empty names are required")
    return Replacement(
        module,
        registry,
        Mechanism.REGISTRY_OVERRIDE,
        mechanism_options={"names": normalized},
    )


def wrap(
    target: str,
    replacement: str,
    *,
    aliases: Iterable[str] = (),
    runtime_condition: Optional[str] = None,
) -> Replacement:
    """Declare a wrapper receiving ``(original, group_options)``."""

    return Replacement(
        target,
        replacement,
        Mechanism.WRAPPER,
        tuple(aliases),
        runtime_condition,
    )


def _member_name(target: str) -> str:
    parts = target.split(".")
    tail = parts[-1]
    if tail in {"forward", "backward", "apply", "__call__"} and len(parts) > 1:
        tail = f"{parts[-2]}_{tail}"
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", tail).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_") or "member"
    return value


def _replacement_ids(group_id: str, items: Tuple[Replacement, ...]) -> Tuple[str, ...]:
    names = [_member_name(item.target) for item in items]
    counts = {name: names.count(name) for name in names}
    result = []
    used = set()
    for name, item in zip(names, items):
        if counts[name] > 1:
            suffix = hashlib.sha256(item.target.encode("utf-8")).hexdigest()[:8]
            name = f"{name}_{suffix}"
        replacement_id = f"{group_id}.{name}"
        if replacement_id in used:
            index = 2
            while f"{replacement_id}_{index}" in used:
                index += 1
            replacement_id = f"{replacement_id}_{index}"
        used.add(replacement_id)
        result.append(replacement_id)
    return tuple(result)


def group(
    group_id: str,
    *items: Replacement,
    depends_on: Iterable[str] = (),
    compatibility_check: Optional[str] = None,
    registry: Registry = default_registry,
) -> Optimization:
    """Declare and register one atomic optimization group."""

    if not group_id or not items:
        raise ValueError("group_id and at least one replacement are required")
    replacements = tuple(items)
    if any(not isinstance(item, Replacement) for item in replacements):
        raise TypeError(
            "group members must be created by a TurboPhysAI declaration helper"
        )
    replacement_ids = _replacement_ids(group_id, replacements)
    specs = tuple(
        ReplacementSpec(
            replacement_id=replacement_id,
            mechanism=item.mechanism,
            target=item.target,
            replacement=item.replacement,
            aliases=item.aliases,
            runtime_condition=item.runtime_condition,
            mechanism_options=item.mechanism_options,
        )
        for replacement_id, item in zip(replacement_ids, replacements)
    )
    declaration = Optimization(
        group_id,
        replacements,
        specs,
        OptimizationGroup(
            group_id,
            replacement_ids,
            depends_on=tuple(depends_on),
            compatibility_check=compatibility_check,
        ),
    )
    declaration.register(registry)
    return declaration


__all__ = [
    "Optimization",
    "Replacement",
    "group",
    "import_alias",
    "optional_import",
    "registry_override",
    "replace",
    "replace_import",
    "wrap",
]
