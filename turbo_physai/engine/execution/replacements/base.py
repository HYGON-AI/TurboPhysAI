# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ...contracts import Mechanism, ReplacementSpec, RestoreResult, RestoreStatus


class HandlerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedAttribute:
    path: str
    parent: Any
    attribute: str
    original: Any
    is_mapping: bool = False


@dataclass(frozen=True)
class PreparedReplacement:
    spec: ReplacementSpec
    targets: Tuple[ResolvedAttribute, ...]
    replacement: Any
    implementation: Any = None
    runtime_condition: Any = None

    def __post_init__(self) -> None:
        if self.implementation is None:
            object.__setattr__(self, "implementation", self.replacement)


def resolve_attribute(path: str, *, import_missing: bool = True) -> ResolvedAttribute:
    parts = path.split(".")
    if len(parts) < 2 or any(not item for item in parts):
        raise HandlerError(f"invalid target path: {path}")

    module = None
    module_length = 0
    for index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = (
                importlib.import_module(module_name)
                if import_missing
                else __imported(module_name)
            )
        except (ImportError, ModuleNotFoundError):
            continue
        if module is not None:
            module_length = index
            break
    if module is None:
        raise HandlerError(f"target module cannot be resolved: {path}")

    parent = module
    for item in parts[module_length:-1]:
        if isinstance(parent, Mapping):
            if item not in parent:
                raise HandlerError(f"target parent cannot be resolved: {path}")
            parent = parent[item]
        else:
            if not hasattr(parent, item):
                raise HandlerError(f"target parent cannot be resolved: {path}")
            parent = getattr(parent, item)
    attribute = parts[-1]
    is_mapping = isinstance(parent, Mapping)
    exists = attribute in parent if is_mapping else hasattr(parent, attribute)
    if not exists:
        raise HandlerError(f"target does not exist: {path}")
    original = parent[attribute] if is_mapping else getattr(parent, attribute)
    return ResolvedAttribute(path, parent, attribute, original, is_mapping)


def resolve_replacement(path: str, *, import_missing: bool = True) -> Any:
    """Resolve a fixed replacement module or object path."""

    try:
        return resolve_attribute(path, import_missing=import_missing).original
    except HandlerError as attribute_error:
        try:
            module = (
                importlib.import_module(path)
                if import_missing
                else __imported(path)
            )
        except ImportError as exc:
            raise HandlerError(
                f"replacement cannot be resolved: {path}: {exc}"
            ) from exc
        if module is not None:
            return module
        raise HandlerError(
            f"replacement cannot be resolved: {path}"
        ) from attribute_error


def __imported(module_name: str) -> Any:
    import sys

    return sys.modules.get(module_name)


def set_attribute(target: ResolvedAttribute, value: Any) -> None:
    if target.is_mapping:
        target.parent[target.attribute] = value
    else:
        setattr(target.parent, target.attribute, value)


def resolve_targets(
    spec: ReplacementSpec,
    *,
    import_missing: bool = True,
) -> Tuple[ResolvedAttribute, ...]:
    """Resolve a target and aliases and verify they share one object."""

    targets = tuple(
        resolve_attribute(path, import_missing=import_missing)
        for path in (spec.target,) + spec.aliases
    )
    if any(target.original is not targets[0].original for target in targets[1:]):
        raise HandlerError(
            f"aliases do not reference the same original object: {spec.replacement_id}"
        )
    return targets


def prepare_runtime_condition(
    spec: ReplacementSpec,
    original: Any,
    implementation: Any,
    *,
    import_missing: bool = True,
) -> Tuple[Any, Optional[Any]]:
    """Build a call-time dispatcher when ``runtime_condition`` is declared."""

    if spec.runtime_condition is None:
        return implementation, None
    if isinstance(original, type):
        raise HandlerError(
            f"runtime_condition does not support class targets: {spec.replacement_id}"
        )
    if not callable(original) or not callable(implementation):
        raise HandlerError(
            f"runtime_condition requires callable target and replacement: "
            f"{spec.replacement_id}"
        )
    try:
        condition = resolve_replacement(
            spec.runtime_condition,
            import_missing=import_missing,
        )
    except HandlerError as exc:
        raise HandlerError(
            f"runtime_condition cannot be resolved: {spec.runtime_condition}: "
            f"{exc}"
        ) from exc
    if not callable(condition) or isinstance(condition, type):
        raise HandlerError(
            f"runtime condition must be a non-class callable: {spec.replacement_id}"
        )

    @functools.wraps(original)
    def conditional_dispatch(*args: Any, **kwargs: Any) -> Any:
        decision = condition(*args, **kwargs)
        if not isinstance(decision, bool):
            raise TypeError(
                "runtime_condition must return bool: "
                f"{spec.runtime_condition}, actual={type(decision).__name__}"
            )
        if decision:
            return implementation(*args, **kwargs)
        return original(*args, **kwargs)

    conditional_dispatch.__turbo_physai_runtime_condition__ = spec.runtime_condition
    conditional_dispatch.__turbo_physai_optimized__ = implementation
    return conditional_dispatch, condition


class MechanismHandler:
    mechanism: Mechanism

    def prepare(
        self,
        spec: ReplacementSpec,
        options: Mapping[str, Any],
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        raise NotImplementedError

    def snapshot(self, prepared: PreparedReplacement) -> Tuple[ResolvedAttribute, ...]:
        return prepared.targets

    def apply(self, prepared: PreparedReplacement) -> Tuple[str, ...]:
        changed: List[str] = []
        for target in prepared.targets:
            set_attribute(target, prepared.replacement)
            changed.append(target.path)
        return tuple(changed)

    def restore(
        self, snapshot: Sequence[ResolvedAttribute]
    ) -> Tuple[RestoreResult, ...]:
        results = []
        for target in reversed(tuple(snapshot)):
            try:
                set_attribute(target, target.original)
                actual = (
                    target.parent[target.attribute]
                    if target.is_mapping
                    else getattr(target.parent, target.attribute)
                )
                if actual is not target.original:
                    raise HandlerError(
                        "restore verification failed: restored object identity "
                        f"does not match snapshot: {target.path}"
                    )
                results.append(RestoreResult(target.path, RestoreStatus.RESTORED))
            except Exception as exc:
                results.append(
                    RestoreResult(target.path, RestoreStatus.FAILED, str(exc))
                )
        return tuple(results)
