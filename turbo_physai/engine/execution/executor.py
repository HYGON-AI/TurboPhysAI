# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from ..contracts import (
    ExecutionStatus,
    GroupExecutionResult,
    ReplacementResult,
    PreparedExecution,
    RestoreResult,
    RestoreStatus,
)


@dataclass(frozen=True)
class ExecutionOutcome:
    groups: Tuple[GroupExecutionResult, ...]
    terminal_error: Optional[str] = None
    rollback_failed: bool = False
    applied_snapshots: Tuple[Tuple[str, Any, Any], ...] = ()


class Executor:
    def execute(
        self,
        prepared_execution: PreparedExecution,
        *,
        prepared_groups: Mapping,
    ) -> ExecutionOutcome:
        prepared_groups_by_id = {
            group.group_id: group for group in prepared_execution.groups
        }
        completed_groups = {}
        results: List[GroupExecutionResult] = []
        terminal_error = None
        rollback_failed = False
        applied_snapshots = []

        for index, group_id in enumerate(prepared_execution.execution_order):
            prepared_group = prepared_groups_by_id[group_id]
            unavailable_dependencies = tuple(
                dependency
                for dependency in prepared_group.depends_on
                if dependency not in completed_groups
                or completed_groups[dependency].status != ExecutionStatus.APPLIED
            )
            if unavailable_dependencies:
                result = GroupExecutionResult(
                    group_id=group_id,
                    status=ExecutionStatus.NOT_STARTED,
                    forced=prepared_group.forced,
                    error=(
                        "dependencies not applied: "
                        + ", ".join(unavailable_dependencies)
                    ),
                )
                results.append(result)
                completed_groups[group_id] = result
                continue
            result, group_snapshots = self._execute_group(
                group_id,
                prepared_group.members,
                prepared_group,
                prepared_groups.get(group_id, ()),
            )
            results.append(result)
            applied_snapshots.extend(group_snapshots)
            completed_groups[group_id] = result
            if result.status == ExecutionStatus.APPLIED:
                continue
            group_rollback_failed = any(
                restore.status == RestoreStatus.FAILED
                for restore in result.rollback_results
            )
            rollback_failed = rollback_failed or group_rollback_failed
            if group_rollback_failed:
                terminal_error = (
                    result.error or f"OptimizationGroup execution failed: {group_id}"
                )
                for remaining in prepared_execution.execution_order[index + 1 :]:
                    remaining_group = prepared_groups_by_id[remaining]
                    results.append(
                        GroupExecutionResult(
                            group_id=remaining,
                            status=ExecutionStatus.NOT_STARTED,
                            forced=remaining_group.forced,
                            error="not started after terminal execution failure",
                        )
                    )
                break
        return ExecutionOutcome(
            tuple(results),
            terminal_error,
            rollback_failed,
            tuple(applied_snapshots),
        )

    @staticmethod
    def restore_applied(
        snapshots: Sequence[Tuple[str, Any, Any]],
    ) -> Tuple[RestoreResult, ...]:
        """Restore successfully applied units retained for a temporary phase."""

        results = []
        for _, handler, snapshot in reversed(tuple(snapshots)):
            try:
                results.extend(handler.restore(snapshot))
            except Exception as exc:
                results.append(
                    RestoreResult(
                        "temporary import compatibility",
                        RestoreStatus.FAILED,
                        str(exc),
                    )
                )
        return tuple(results)

    def _execute_group(
        self,
        group_id: str,
        replacement_ids: Sequence[str],
        prepared_group,
        prepared_units,
    ) -> Tuple[GroupExecutionResult, Tuple[Tuple[str, Any, Any], ...]]:
        prepared = list(prepared_units)
        unit_results: List[ReplacementResult] = []
        snapshots = []
        try:
            if tuple(item[0] for item in prepared) != tuple(replacement_ids):
                raise RuntimeError(
                    f"prepared OptimizationGroup does not match PreparedExecution: {group_id}"
                )
            # Snapshot the complete atomic group before the first mutation.
            for replacement_id, handler, prepared_replacement in prepared:
                snapshots.append(
                    (
                        replacement_id,
                        handler,
                        handler.snapshot(prepared_replacement),
                    )
                )
        except Exception as exc:
            failed_index = min(len(prepared), len(replacement_ids) - 1)
            unit_results = [
                ReplacementResult(
                    replacement_id,
                    (
                        ExecutionStatus.FAILED
                        if index == failed_index
                        else ExecutionStatus.NOT_STARTED
                    ),
                    error=str(exc) if index == failed_index else None,
                )
                for index, replacement_id in enumerate(replacement_ids)
            ]
            return (
                GroupExecutionResult(
                    group_id,
                    ExecutionStatus.FAILED,
                    tuple(unit_results),
                    forced=prepared_group.forced,
                    error=str(exc),
                ),
                (),
            )

        failure = None
        for replacement_id, handler, prepared_replacement in prepared:
            started = time.perf_counter()
            try:
                changed = handler.apply(prepared_replacement)
                unit_results.append(
                    ReplacementResult(
                        replacement_id,
                        ExecutionStatus.APPLIED,
                        changed_targets=changed,
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )
            except Exception as exc:
                failure = (replacement_id, exc)
                unit_results.append(
                    ReplacementResult(
                        replacement_id,
                        ExecutionStatus.FAILED,
                        error=str(exc),
                        duration_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )
                break

        if failure is None:
            return (
                GroupExecutionResult(
                    group_id,
                    ExecutionStatus.APPLIED,
                    tuple(unit_results),
                    forced=prepared_group.forced,
                ),
                tuple(
                    (replacement_id, handler, snapshot)
                    for replacement_id, handler, snapshot in snapshots
                ),
            )

        completed = len(unit_results)
        for replacement_id in replacement_ids[completed:]:
            unit_results.append(ReplacementResult(replacement_id, ExecutionStatus.NOT_STARTED))
        restore_results = []
        for _, handler, snapshot in reversed(snapshots):
            try:
                restore_results.extend(handler.restore(snapshot))
            except Exception as exc:
                restore_results.append(
                    RestoreResult(group_id, RestoreStatus.FAILED, str(exc))
                )
        rollback_ok = all(
            item.status != RestoreStatus.FAILED for item in restore_results
        )
        rolled_units = []
        for unit in unit_results:
            if unit.status == ExecutionStatus.APPLIED:
                rolled_units.append(
                    ReplacementResult(
                        unit.replacement_id,
                        ExecutionStatus.ROLLED_BACK,
                        unit.changed_targets,
                        unit.error,
                        unit.duration_ms,
                    )
                )
            else:
                rolled_units.append(unit)
        return (
            GroupExecutionResult(
                group_id,
                ExecutionStatus.ROLLED_BACK if rollback_ok else ExecutionStatus.FAILED,
                tuple(rolled_units),
                tuple(restore_results),
                prepared_group.forced,
                f"{failure[0]}: {failure[1]}",
            ),
            (),
        )
