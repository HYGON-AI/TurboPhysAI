# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import re
import uuid
from dataclasses import replace as dataclass_replace
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Sequence, Tuple, Union

from .checking.context import detect_context
from .errors import (
    OptimizationRollbackError,
    OptimizationConfigError,
)
from .execution.executor import Executor
from .contracts import (
    CheckResult,
    CheckStatus,
    Decision,
    ExecutionStatus,
    OptimizationConfig,
    OptimizationReport,
    PreparedGroup,
    PreparedExecution,
)
from .checking.ordering import Preparation
from .definitions import (
    Optimization,
    group,
    import_alias,
    optional_import,
    registry_override,
    replace,
    replace_import,
    wrap,
)
from .definitions.registry import Registry, default_registry as _default_registry
from .execution.report import build_report, emit_report
from .execution.replacements import default_handlers
from .config.loader import OptimizationConfigCatalog, load_optimization_config, resolve_optimization_config_path


PathLike = Union[str, os.PathLike]


_IMPORT_COMPATIBILITY_MECHANISMS = {
    "import_replace",
    "import_alias",
    "optional_import",
    "registry_override",
}


_apply_lock = Lock()
_apply_called = False
_RUN_ID_ENV = "TURBO_PHYSAI_RUN_ID"
_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def _resolve_run_id() -> str:
    """Return the shared CLI-launch ID or create one for direct API use."""

    configured = os.environ.get(_RUN_ID_ENV)
    if configured is None:
        return uuid.uuid4().hex
    if _RUN_ID_PATTERN.fullmatch(configured) is None:
        raise OptimizationConfigError(
            f"{_RUN_ID_ENV} must be a 32-character lowercase hexadecimal ID"
        )
    return configured


def _claim_apply() -> None:
    """Allow exactly one apply attempt in the current process."""

    global _apply_called
    with _apply_lock:
        if _apply_called:
            raise OptimizationConfigError(
                "turbo_physai.apply() may only be called once per process; "
                "restart the process before applying an OptimizationConfig again"
            )
        _apply_called = True


def _validate_group_ids(
    config: OptimizationConfig,
    values: Sequence[str],
    *,
    name: str,
) -> Tuple[str, ...]:
    if isinstance(values, str):
        raise OptimizationConfigError(f"{name} must be a sequence of Group IDs")
    normalized = tuple(dict.fromkeys(values))
    if any(not isinstance(group_id, str) or not group_id for group_id in normalized):
        raise OptimizationConfigError(
            f"{name} must contain non-empty Group IDs"
        )
    enabled = {entry.id for entry in config.optimization_groups if entry.enabled}
    unavailable = tuple(group_id for group_id in normalized if group_id not in enabled)
    if unavailable:
        raise OptimizationConfigError(
            f"{name} must reference enabled OptimizationGroups: "
            + ", ".join(unavailable)
        )
    return normalized


def _apply_group_overrides(
    config: OptimizationConfig,
    registry: Registry,
    force_groups: Sequence[str],
    disable_groups: Sequence[str],
    accuracy_mode: bool = False,
) -> Tuple[OptimizationConfig, Tuple[str, ...], Dict[str, str]]:
    forced = _validate_group_ids(config, force_groups, name="force_groups")
    disabled = _validate_group_ids(config, disable_groups, name="disable_groups")
    overlap = tuple(group_id for group_id in forced if group_id in disabled)
    if overlap:
        raise OptimizationConfigError(
            "OptimizationGroups cannot be both forced and disabled: "
            + ", ".join(overlap)
        )

    disabled_reasons: Dict[str, str] = {
        group_id: "disabled_by_user" for group_id in disabled
    }
    if not accuracy_mode:
        for entry in config.optimization_groups:
            if entry.accuracy_only and entry.id not in disabled_reasons:
                disabled_reasons[entry.id] = "accuracy_only_group"
        accuracy_forced = tuple(
            group_id
            for group_id in forced
            if disabled_reasons.get(group_id) == "accuracy_only_group"
        )
        if accuracy_forced:
            raise OptimizationConfigError(
                "--force-group cannot enable accuracy-only OptimizationGroups "
                "while accuracy mode is off: " + ", ".join(accuracy_forced)
            )

    enabled_ids = {
        entry.id for entry in config.optimization_groups if entry.enabled
    }
    changed = True
    while changed:
        changed = False
        for group_id in enabled_ids:
            if group_id in disabled_reasons:
                continue
            definition = registry.get_group(group_id)
            if definition and any(
                dependency in disabled_reasons
                for dependency in definition.depends_on
            ):
                disabled_reasons[group_id] = "dependency_disabled"
                changed = True

    unavailable_forced = tuple(
        group_id for group_id in forced if group_id in disabled_reasons
    )
    if unavailable_forced:
        raise OptimizationConfigError(
            "force_groups depend on disabled OptimizationGroups: "
            + ", ".join(unavailable_forced)
        )

    effective = dataclass_replace(
        config,
        optimization_groups=tuple(
            dataclass_replace(entry, enabled=False)
            if entry.id in disabled_reasons
            else entry
            for entry in config.optimization_groups
        ),
    )
    return effective, forced, disabled_reasons


class _Application:
    """One apply attempt, with optional separation at a script's entry guard."""

    def __init__(
        self,
        *,
        optimization_config_path=None,
        registry=None,
        catalog=None,
        model=None,
        force_groups=(),
        disable_groups=(),
        accuracy_mode=False,
    ):
        _claim_apply()
        self.config_path = resolve_optimization_config_path(
            optimization_config_path, model=model
        )
        config = load_optimization_config(self.config_path, catalog=catalog)
        self.registry = registry or _default_registry
        self.config, self.forced, self.disabled_reasons = _apply_group_overrides(
            config, self.registry, force_groups, disable_groups, accuracy_mode
        )
        self.run_id = _resolve_run_id()
        self.handlers = default_handlers()
        self.executor = Executor()
        self.prepared = []
        self.results = []
        self.terminal_error = None

        compatibility_entries = []
        regular_entries = []
        for entry in self.config.optimization_groups:
            definition = self.registry.get_group(entry.id)
            mechanisms = {
                spec.mechanism.value
                for member in (definition.members if definition else ())
                if (spec := self.registry.get_spec(member)) is not None
            }
            if mechanisms & _IMPORT_COMPATIBILITY_MECHANISMS:
                if not mechanisms.issubset(_IMPORT_COMPATIBILITY_MECHANISMS):
                    raise OptimizationConfigError(
                        "import compatibility and runtime replacements must use "
                        f"separate OptimizationGroups: {entry.id}"
                    )
                compatibility_entries.append(entry)
            else:
                regular_entries.append(entry)

        compatibility_ids = {entry.id for entry in compatibility_entries}
        for entry in compatibility_entries:
            definition = self.registry.get_group(entry.id)
            invalid = tuple(
                group_id for group_id in definition.depends_on
                if group_id not in compatibility_ids
            )
            if invalid:
                raise OptimizationConfigError(
                    "import compatibility Group dependencies must also be import "
                    f"compatibility Groups: {entry.id}: {', '.join(invalid)}"
                )
        self.compatibility = dataclass_replace(
            self.config, optimization_groups=tuple(compatibility_entries)
        )
        self.regular = dataclass_replace(
            self.config, optimization_groups=tuple(regular_entries)
        )

    def _prepare_and_apply(self, config, **kwargs):
        preparation = Preparation(self.registry, self.handlers)
        prepared = preparation.prepare(
            run_id=self.run_id,
            config=config,
            environment=self.environment,
            force_groups=self.forced,
            disabled_reasons=self.disabled_reasons,
            **kwargs,
        )
        self.prepared.append(prepared)
        outcome = self.executor.execute(
            prepared,
            prepared_groups=preparation.prepared_groups,
            previous_results=self.results,
        )
        self.results.extend(outcome.groups)
        self.terminal_error = outcome.terminal_error
        return prepared, outcome

    def apply_compatibility(self):
        dependencies = self.config.compatibility.get("dependencies", {})
        names = dependencies.keys() if hasattr(dependencies, "keys") else ()
        self.environment = detect_context(dependency_names=names)
        self.compatibility_ok = True
        if self.compatibility.optimization_groups:
            prepared, outcome = self._prepare_and_apply(self.compatibility)
            self.compatibility_ok = all(
                group.decision in {Decision.APPLY, Decision.SKIP}
                for group in prepared.groups
            ) and all(
                item.status == ExecutionStatus.APPLIED for item in outcome.groups
            ) and not self.terminal_error

    def apply_regular(self, *, group_ids=None, blocked_groups=None, target_key=None):
        if not self.compatibility_ok:
            self._block_regular(
                CheckResult(
                    "import_compatibility.blocked",
                    CheckStatus.FAIL,
                    expected="all import compatibility Groups applied",
                    actual="import compatibility application failed",
                ),
                "import_compatibility_blocked",
            )
        elif not self.terminal_error:
            self._prepare_and_apply(
                self.regular,
                group_ids=group_ids,
                previous_groups={
                    group.group_id: group
                    for stage in self.prepared for group in stage.groups
                },
                blocked_groups=blocked_groups,
                target_key=target_key,
            )

    def _block_regular(self, check, reason, *, group_ids=None, checks=()):
        groups = []
        for entry in self.regular.optimization_groups:
            if group_ids is not None and entry.id not in group_ids:
                continue
            definition = self.registry.get_group(entry.id)
            groups.append(PreparedGroup(
                entry.id,
                definition.depends_on if definition else (),
                definition.members if definition else (),
                (check,),
                Decision.BLOCK,
                reason,
            ))
        self.prepared.append(PreparedExecution(
            self.run_id, self.environment, tuple(groups), (), (), checks
        ))

    def block_pending_after_failure(self, group_ids):
        self._block_regular(
            CheckResult("entrypoint.startup_failed", CheckStatus.FAIL, detail=self.terminal_error),
            "entrypoint_startup_failed",
            group_ids=group_ids,
            checks=self.prepared[-1].checks,
        )

    def report(self, *, log_report):
        prepared = {
            group.group_id: group
            for stage in self.prepared for group in stage.groups
        }
        results = {item.group_id: item for item in self.results}
        ids = [entry.id for entry in self.config.optimization_groups]
        prepared_execution = PreparedExecution(
            self.run_id,
            self.environment,
            tuple(prepared[group_id] for group_id in ids if group_id in prepared),
            tuple(dict.fromkeys(
                conflict for stage in self.prepared for conflict in stage.conflicts
            )),
            tuple(group_id for stage in self.prepared for group_id in stage.execution_order),
            self.prepared[-1].checks,
        )
        report = build_report(
            self.config,
            prepared_execution,
            (results[group_id] for group_id in ids if group_id in results),
            optimization_config_path=str(self.config_path),
            runtime_config_path=os.environ.get("TURBO_PHYSAI_RUNTIME_CONFIG_PATH"),
        )
        if log_report and self.environment.rank == 0:
            emit_report(report)
        if self.terminal_error:
            raise OptimizationRollbackError(self.terminal_error, report=report)
        return report


def apply(
    *,
    optimization_config_path: Optional[PathLike] = None,
    model: Optional[str] = None,
    log_report: bool = False,
    registry: Optional[Registry] = None,
    catalog: Optional[OptimizationConfigCatalog] = None,
    force_groups: Sequence[str] = (),
    disable_groups: Sequence[str] = (),
    accuracy_mode: bool = False,
) -> OptimizationReport:
    """Apply one OptimizationConfig during process startup and report its result.

    Call this entry point once per process, before importing the target model.
    ``model`` selects a packaged model OptimizationConfig when no explicit path is
    provided. ``log_report`` writes the complete OptimizationReport to standard
    output on rank 0. ``disable_groups`` skips the named Groups and their
    dependents for this call only.
    Repeated application and in-process retry are intentionally unsupported.
    Blocked or successfully rolled-back Groups are isolated together with their
    dependents; unrelated Groups continue.  A rollback failure is terminal.
    ``force_groups`` bypasses only overrideable checks for the named enabled Groups.
    ``accuracy_mode`` applies ``accuracy_only`` Groups; when False those Groups and
    their dependents are skipped for this call.
    """

    application = _Application(
        optimization_config_path=optimization_config_path,
        registry=registry,
        catalog=catalog,
        model=model,
        force_groups=force_groups,
        disable_groups=disable_groups,
        accuracy_mode=accuracy_mode,
    )
    application.apply_compatibility()
    application.apply_regular()
    return application.report(log_report=log_report)


__all__ = [
    "Optimization",
    "apply",
    "group",
    "import_alias",
    "optional_import",
    "registry_override",
    "replace",
    "replace_import",
    "wrap",
]
