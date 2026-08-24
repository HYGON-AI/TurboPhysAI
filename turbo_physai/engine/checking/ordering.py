# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from .checker import Checker
from .conflicts import find_group_conflicts
from ..contracts import (
    CheckResult,
    CheckStatus,
    Conflict,
    Decision,
    EnvironmentSnapshot,
    OptimizationConfig,
    PreparedGroup,
    PreparedExecution,
)
from ..definitions.registry import Registry


def _decide(checks, *, force: bool = False):
    force_used = False
    blocking = []
    for check in checks:
        if check.status in {
            CheckStatus.PASS,
            CheckStatus.WARNING,
            CheckStatus.NOT_APPLICABLE,
        }:
            continue
        if force and check.overrideable:
            force_used = True
            continue
        blocking.append(check.code)
    if blocking:
        return (
            Decision.BLOCK,
            "checks_failed:" + ",".join(sorted(set(blocking))),
            False,
        )
    return Decision.APPLY, "checks_passed", force_used


class Preparation:
    def __init__(
        self,
        registry: Registry,
        handlers: Mapping,
    ) -> None:
        self.registry = registry
        self.handlers: Mapping = handlers
        self.checker = Checker(registry, handlers)

    @property
    def prepared_groups(self):
        """Runtime-only prepared objects produced by the latest resolution."""

        return dict(self.checker.prepared_groups)

    def prepare(
        self,
        *,
        run_id: str,
        config: OptimizationConfig,
        environment: EnvironmentSnapshot,
        force_groups: Sequence[str] = (),
        import_missing: bool = True,
    ) -> PreparedExecution:
        self.checker.prepared_groups.clear()
        forced = frozenset(force_groups)
        config_checks = self.checker.check_environment(
            environment, config.compatibility
        )
        enabled_ids = [entry.id for entry in config.optimization_groups if entry.enabled]
        conflicts = find_group_conflicts(self.registry, enabled_ids)
        conflict_by_group: Dict[str, List[Conflict]] = defaultdict(list)
        for conflict in conflicts:
            for group_id in conflict.groups:
                conflict_by_group[group_id].append(conflict)

        order, cycle_groups = self._topological_order(enabled_ids)
        if cycle_groups:
            cycle = Conflict(
                "dependency.cycle",
                tuple(sorted(cycle_groups)),
                detail="OptimizationGroup dependency graph contains a cycle",
            )
            conflicts.append(cycle)
            for group_id in cycle.groups:
                conflict_by_group[group_id].append(cycle)

        prepared_groups: Dict[str, PreparedGroup] = {}
        for entry in config.optimization_groups:
            group = self.registry.get_group(entry.id)
            if not entry.enabled:
                prepared_groups[entry.id] = PreparedGroup(
                    entry.id,
                    (),
                    (),
                    (),
                    Decision.SKIP,
                    "group_disabled",
                )
                continue
            if group is None:
                check = CheckResult(
                    "registry.group_missing",
                    CheckStatus.FAIL,
                    expected=entry.id,
                    detail="OptimizationGroup is not registered",
                )
                prepared_groups[entry.id] = PreparedGroup(
                    entry.id,
                    (),
                    (),
                    (check,),
                    Decision.BLOCK,
                    "group_missing",
                )
                continue
            checks = list(
                self.checker.check_group(
                    group,
                    entry,
                    environment,
                    import_missing=import_missing,
                )
            )
            for conflict in conflict_by_group.get(group.group_id, ()):
                checks.append(
                    CheckResult(
                        conflict.code,
                        CheckStatus.FAIL,
                        expected="no conflict",
                        actual={
                            "groups": conflict.groups,
                            "replacement_ids": conflict.replacement_ids,
                            "target": conflict.target,
                            "replacements": conflict.replacements,
                        },
                        detail=conflict.detail,
                    )
                )
            decision, reason, forced_group = _decide(
                checks=config_checks + tuple(checks),
                force=group.group_id in forced,
            )
            prepared_groups[group.group_id] = PreparedGroup(
                group.group_id,
                group.depends_on,
                group.members,
                tuple(checks),
                decision,
                reason,
                forced_group,
            )

        # A selected group cannot execute when one of its required dependencies is blocked.
        changed = True
        while changed:
            changed = False
            for group_id in order:
                current = prepared_groups.get(group_id)
                group = self.registry.get_group(group_id)
                if (
                    current is None
                    or group is None
                    or current.decision != Decision.APPLY
                ):
                    continue
                blocked_dependencies = [
                    dependency
                    for dependency in group.depends_on
                    if dependency not in prepared_groups
                    or prepared_groups[dependency].decision != Decision.APPLY
                ]
                if blocked_dependencies:
                    dependency_check = CheckResult(
                        "dependency.blocked",
                        CheckStatus.FAIL,
                        expected=group.depends_on,
                        actual=tuple(blocked_dependencies),
                        detail="dependency OptimizationGroup is not applicable",
                    )
                    prepared_groups[group_id] = PreparedGroup(
                        current.group_id,
                        current.depends_on,
                        current.members,
                        current.checks + (dependency_check,),
                        Decision.BLOCK,
                        "dependency_blocked",
                        current.forced,
                    )
                    changed = True

        stable_groups = tuple(
            prepared_groups[entry.id] for entry in config.optimization_groups
        )
        execution_order = tuple(
            group_id
            for group_id in order
            if group_id in prepared_groups
            and prepared_groups[group_id].decision == Decision.APPLY
        )
        return PreparedExecution(
            run_id=run_id,
            environment=environment,
            groups=stable_groups,
            conflicts=tuple(conflicts),
            execution_order=execution_order,
            checks=config_checks,
        )

    def _topological_order(
        self, enabled_ids: Sequence[str]
    ) -> Tuple[List[str], Set[str]]:
        enabled = set(enabled_ids)
        declaration_index = {
            group_id: index for index, group_id in enumerate(enabled_ids)
        }
        edges: Dict[str, Set[str]] = {group_id: set() for group_id in enabled_ids}
        indegree = {group_id: 0 for group_id in enabled_ids}
        for group_id in enabled_ids:
            group = self.registry.get_group(group_id)
            if group is None:
                continue
            for predecessor in group.depends_on:
                if predecessor in enabled and group_id not in edges[predecessor]:
                    edges[predecessor].add(group_id)
                    indegree[group_id] += 1
        ready = [group_id for group_id in enabled_ids if indegree[group_id] == 0]
        ordered: List[str] = []
        while ready:
            group_id = ready.pop(0)
            ordered.append(group_id)
            for successor in sorted(
                edges[group_id], key=declaration_index.__getitem__
            ):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=declaration_index.__getitem__)
        cycle = {group_id for group_id, degree in indegree.items() if degree > 0}
        ordered.extend(group_id for group_id in enabled_ids if group_id in cycle)
        return ordered, cycle
