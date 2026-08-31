# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ...contracts import Mechanism, ReplacementSpec, RestoreResult, RestoreStatus
from .base import HandlerError, MechanismHandler, PreparedReplacement


@dataclass(frozen=True)
class ImportAliasSnapshot:
    module: Any
    alias: str
    existed: bool
    original: Any


class ImportAliasHandler(MechanismHandler):
    mechanism = Mechanism.IMPORT_ALIAS

    def prepare(
        self,
        spec: ReplacementSpec,
        options: Mapping[str, Any],
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        del options
        module_name, _, alias = spec.target.rpartition(".")
        source_module, _, source = spec.replacement.rpartition(".")
        if not module_name or module_name != source_module:
            raise HandlerError(
                f"import alias source and alias must belong to one module: "
                f"{spec.replacement_id}"
            )
        module = (
            importlib.import_module(module_name)
            if import_missing
            else __import__("sys").modules.get(module_name)
        )
        if module is None or not hasattr(module, source):
            raise HandlerError(f"import alias source does not exist: {spec.replacement}")
        source_value = getattr(module, source)
        if hasattr(module, alias) and getattr(module, alias) is not source_value:
            raise HandlerError(
                f"import alias already exists with another value: {spec.target}"
            )
        return PreparedReplacement(spec, (), (module, alias, source_value))

    def snapshot(self, prepared: PreparedReplacement) -> ImportAliasSnapshot:
        module, alias, _ = prepared.replacement
        return ImportAliasSnapshot(
            module,
            alias,
            hasattr(module, alias),
            getattr(module, alias, None),
        )

    def apply(self, prepared: PreparedReplacement) -> Tuple[str, ...]:
        module, alias, source_value = prepared.replacement
        if hasattr(module, alias) and getattr(module, alias) is not source_value:
            raise HandlerError(
                f"import alias changed after preparation: {prepared.spec.target}"
            )
        setattr(module, alias, source_value)
        return (prepared.spec.target,)

    def restore(self, snapshot: ImportAliasSnapshot) -> Tuple[RestoreResult, ...]:
        try:
            if snapshot.existed:
                setattr(snapshot.module, snapshot.alias, snapshot.original)
            elif hasattr(snapshot.module, snapshot.alias):
                delattr(snapshot.module, snapshot.alias)
            return (
                RestoreResult(
                    f"{snapshot.module.__name__}.{snapshot.alias}",
                    RestoreStatus.RESTORED,
                ),
            )
        except Exception as exc:
            return (
                RestoreResult(
                    f"{snapshot.module.__name__}.{snapshot.alias}",
                    RestoreStatus.FAILED,
                    str(exc),
                ),
            )
