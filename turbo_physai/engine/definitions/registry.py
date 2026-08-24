# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional

from ..contracts import OptimizationGroup, ReplacementSpec


class Registry:
    """Explicit registry for declarative ReplacementSpecs and OptimizationGroups."""

    def __init__(self) -> None:
        self._specs: Dict[str, ReplacementSpec] = {}
        self._groups: Dict[str, OptimizationGroup] = {}

    def register_spec(self, spec: ReplacementSpec) -> ReplacementSpec:
        self._register_unique(self._specs, spec.replacement_id, spec, "ReplacementSpec")
        return spec

    def register_group(self, group: OptimizationGroup) -> OptimizationGroup:
        self._register_unique(self._groups, group.group_id, group, "OptimizationGroup")
        return group

    @staticmethod
    def _register_unique(
        target: Dict[str, Any], key: str, value: Any, kind: str
    ) -> None:
        if not key:
            raise ValueError(f"{kind} id cannot be empty")
        if key in target:
            raise ValueError(f"duplicate {kind} id: {key}")
        target[key] = value

    def get_spec(self, replacement_id: str) -> Optional[ReplacementSpec]:
        return self._specs.get(replacement_id)

    def get_group(self, group_id: str) -> Optional[OptimizationGroup]:
        return self._groups.get(group_id)

    @property
    def specs(self) -> Mapping[str, ReplacementSpec]:
        return dict(self._specs)

    @property
    def groups(self) -> Mapping[str, OptimizationGroup]:
        return dict(self._groups)


default_registry = Registry()
