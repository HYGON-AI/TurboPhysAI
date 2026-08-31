# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, List, Mapping, Tuple

from ...contracts import Mechanism, ReplacementSpec, RestoreResult, RestoreStatus
from .base import HandlerError, MechanismHandler, PreparedReplacement, resolve_replacement


@dataclass(frozen=True)
class ImportSnapshot:
    modules: Tuple[Tuple[str, bool, Any], ...]
    parent_links: Tuple[Tuple[Any, str, bool, Any], ...]


class ImportReplaceHandler(MechanismHandler):
    mechanism = Mechanism.IMPORT_REPLACE

    def prepare(
        self,
        spec: ReplacementSpec,
        options: Mapping[str, Any],
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        del options
        if spec.runtime_condition is not None:
            raise HandlerError(
                f"runtime_condition does not support import replacement: "
                f"{spec.replacement_id}"
            )
        replacement = resolve_replacement(
            spec.replacement, import_missing=import_missing
        )
        if not isinstance(replacement, ModuleType):
            raise HandlerError(
                f"import replacement must be a module: {spec.replacement_id}"
            )
        return PreparedReplacement(spec, (), replacement)

    def snapshot(self, prepared: PreparedReplacement) -> ImportSnapshot:
        names = []
        parts = prepared.spec.target.split(".")
        for index in range(1, len(parts) + 1):
            name = ".".join(parts[:index])
            names.append((name, name in sys.modules, sys.modules.get(name)))
        links = []
        for index in range(1, len(parts)):
            parent = sys.modules.get(".".join(parts[:index]))
            if parent is not None:
                child = parts[index]
                links.append(
                    (
                        parent,
                        child,
                        hasattr(parent, child),
                        getattr(parent, child, None),
                    )
                )
        return ImportSnapshot(tuple(names), tuple(links))

    def apply(self, prepared: PreparedReplacement) -> Tuple[str, ...]:
        parts = prepared.spec.target.split(".")
        for index in range(1, len(parts)):
            name = ".".join(parts[:index])
            if name not in sys.modules:
                package = ModuleType(name)
                package.__package__ = name
                package.__path__ = []
                sys.modules[name] = package
            if index > 1:
                parent = sys.modules[".".join(parts[: index - 1])]
                setattr(parent, parts[index - 1], sys.modules[name])
        sys.modules[prepared.spec.target] = prepared.replacement
        if len(parts) > 1:
            setattr(sys.modules[".".join(parts[:-1])], parts[-1], prepared.replacement)
        return (prepared.spec.target,)

    def restore(self, snapshot: ImportSnapshot) -> Tuple[RestoreResult, ...]:
        results: List[RestoreResult] = []
        for parent, child, existed, original in reversed(snapshot.parent_links):
            path = f"{getattr(parent, '__name__', '<module>')}.{child}"
            try:
                if existed:
                    setattr(parent, child, original)
                    if getattr(parent, child, None) is not original:
                        raise HandlerError(
                            "restore verification failed: restored object identity "
                            f"does not match snapshot: {path}"
                        )
                elif hasattr(parent, child):
                    delattr(parent, child)
                if not existed and hasattr(parent, child):
                    raise HandlerError(
                        f"restore verification failed: attribute still exists: {path}"
                    )
                results.append(RestoreResult(path, RestoreStatus.RESTORED))
            except Exception as exc:
                results.append(RestoreResult(path, RestoreStatus.FAILED, str(exc)))
        for name, existed, original in reversed(snapshot.modules):
            try:
                if existed:
                    sys.modules[name] = original
                    if sys.modules.get(name) is not original:
                        raise HandlerError(
                            "restore verification failed: restored module identity "
                            f"does not match snapshot: {name}"
                        )
                else:
                    sys.modules.pop(name, None)
                    if name in sys.modules:
                        raise HandlerError(
                            f"restore verification failed: module still exists: {name}"
                        )
                results.append(RestoreResult(name, RestoreStatus.RESTORED))
            except Exception as exc:
                results.append(RestoreResult(name, RestoreStatus.FAILED, str(exc)))
        return tuple(results)
