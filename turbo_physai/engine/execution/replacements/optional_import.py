# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Tuple

from ...contracts import Mechanism, ReplacementSpec, RestoreResult, RestoreStatus
from .base import HandlerError, MechanismHandler, PreparedReplacement


@dataclass(frozen=True)
class OptionalImportSnapshot:
    module_name: str
    existed: bool
    original: Any
    parent: Any
    child: str
    parent_had_child: bool
    parent_original: Any


class OptionalImportHandler(MechanismHandler):
    mechanism = Mechanism.OPTIONAL_IMPORT

    def prepare(
        self,
        spec: ReplacementSpec,
        options: Mapping[str, Any],
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        del options, import_missing
        if spec.target != spec.replacement or spec.runtime_condition is not None:
            raise HandlerError(
                f"invalid optional import declaration: {spec.replacement_id}"
            )
        return PreparedReplacement(spec, (), spec.target)

    def snapshot(self, prepared: PreparedReplacement) -> OptionalImportSnapshot:
        module_name = prepared.replacement
        parent_name, _, child = module_name.rpartition(".")
        parent = sys.modules.get(parent_name)
        return OptionalImportSnapshot(
            module_name,
            module_name in sys.modules,
            sys.modules.get(module_name),
            parent,
            child,
            parent is not None and hasattr(parent, child),
            getattr(parent, child, None) if parent is not None else None,
        )

    def apply(self, prepared: PreparedReplacement) -> Tuple[str, ...]:
        module_name = prepared.replacement
        current = sys.modules.get(module_name)
        if current is not None:
            return (module_name,)
        placeholder = ModuleType(module_name)
        placeholder.__package__ = module_name.rpartition(".")[0]
        placeholder.__turbo_physai_optional_import__ = True
        sys.modules[module_name] = placeholder
        parent_name, _, child = module_name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child, placeholder)
        return (module_name,)

    def restore(self, snapshot: OptionalImportSnapshot) -> Tuple[RestoreResult, ...]:
        try:
            if snapshot.existed:
                sys.modules[snapshot.module_name] = snapshot.original
            else:
                sys.modules.pop(snapshot.module_name, None)
            if snapshot.parent is not None:
                if snapshot.parent_had_child:
                    setattr(snapshot.parent, snapshot.child, snapshot.parent_original)
                elif hasattr(snapshot.parent, snapshot.child):
                    delattr(snapshot.parent, snapshot.child)
            return (RestoreResult(snapshot.module_name, RestoreStatus.RESTORED),)
        except Exception as exc:
            return (
                RestoreResult(
                    snapshot.module_name, RestoreStatus.FAILED, str(exc)
                ),
            )
