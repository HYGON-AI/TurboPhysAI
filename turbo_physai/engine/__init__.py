# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import sys
import uuid
from dataclasses import replace as dataclass_replace
from pathlib import Path
from threading import Lock
from typing import Optional, Sequence, Tuple, Union

from .checking.context import detect_context
from .errors import (
    OptimizationRollbackError,
    OptimizationConfigError,
)
from .execution.executor import ExecutionOutcome, Executor
from .contracts import (
    CheckResult,
    CheckStatus,
    Decision,
    ExecutionStatus,
    OptimizationConfig,
    OptimizationReport,
    PreparedGroup,
    ReportArtifacts,
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
from .execution.report import build_report, report_paths, write_report
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


def _validate_force_groups(
    config: OptimizationConfig,
    force_groups: Sequence[str],
) -> Tuple[str, ...]:
    if isinstance(force_groups, str):
        raise OptimizationConfigError(
            "force_groups must be a sequence of Group IDs"
        )
    normalized = tuple(dict.fromkeys(force_groups))
    if any(not isinstance(group_id, str) or not group_id for group_id in normalized):
        raise OptimizationConfigError(
            "force_groups must contain non-empty Group IDs"
        )
    enabled = {entry.id for entry in config.optimization_groups if entry.enabled}
    unavailable = tuple(group_id for group_id in normalized if group_id not in enabled)
    if unavailable:
        raise OptimizationConfigError(
            "force_groups must reference enabled OptimizationGroups: "
            + ", ".join(unavailable)
        )
    return normalized


def _resolve(
    *,
    optimization_config_path: Optional[PathLike],
    registry: Optional[Registry],
    catalog: Optional[OptimizationConfigCatalog],
    force_groups: Sequence[str],
    restore_imports: bool,
) -> Tuple[
    OptimizationConfig,
    PreparedExecution,
    dict,
    Path,
    ExecutionOutcome,
    PreparedExecution,
]:
    resolved_config_path = resolve_optimization_config_path(
        optimization_config_path
    )
    config = load_optimization_config(resolved_config_path, catalog=catalog)
    dependencies = config.compatibility.get("dependencies", {})
    dependency_names = dependencies.keys() if hasattr(dependencies, "keys") else ()
    context = detect_context(dependency_names=dependency_names)
    active_registry = registry or _default_registry
    handlers = default_handlers()
    forced = _validate_force_groups(config, force_groups)
    before_modules = dict(sys.modules) if restore_imports else {}
    compatibility_outcome = ExecutionOutcome(())
    try:
        run_id = uuid.uuid4().hex
        compatibility_entries = []
        regular_entries = []
        compatibility_ids = set()
        for entry in config.optimization_groups:
            group_definition = active_registry.get_group(entry.id)
            mechanisms = {
                active_registry.get_spec(member).mechanism.value
                for member in (group_definition.members if group_definition else ())
                if active_registry.get_spec(member) is not None
            }
            compatibility = bool(mechanisms & _IMPORT_COMPATIBILITY_MECHANISMS)
            if compatibility and not mechanisms.issubset(
                _IMPORT_COMPATIBILITY_MECHANISMS
            ):
                raise OptimizationConfigError(
                    "import compatibility and runtime replacements must use "
                    f"separate OptimizationGroups: {entry.id}"
                )
            if compatibility:
                compatibility_entries.append(entry)
                compatibility_ids.add(entry.id)
            else:
                regular_entries.append(entry)

        for group_id in compatibility_ids:
            definition = active_registry.get_group(group_id)
            invalid = tuple(
                dependency
                for dependency in definition.depends_on
                if dependency not in compatibility_ids
            )
            if invalid:
                raise OptimizationConfigError(
                    "import compatibility Group dependencies must also be import "
                    f"compatibility Groups: {group_id}: {', '.join(invalid)}"
                )

        compatibility_prepared = None
        compatibility_runtime = {}
        if compatibility_entries:
            compatibility_config = dataclass_replace(
                config, optimization_groups=tuple(compatibility_entries)
            )
            compatibility_preparation = Preparation(active_registry, handlers)
            compatibility_prepared = compatibility_preparation.prepare(
                run_id=run_id,
                config=compatibility_config,
                environment=context,
                force_groups=tuple(
                    group_id for group_id in forced if group_id in compatibility_ids
                ),
                import_missing=True,
            )
            compatibility_runtime = compatibility_preparation.prepared_groups
            compatibility_outcome = Executor().execute(
                compatibility_prepared,
                prepared_groups=compatibility_runtime,
            )

        compatibility_checks_ok = all(
            group.decision in {Decision.APPLY, Decision.SKIP}
            for group in (
                compatibility_prepared.groups if compatibility_prepared else ()
            )
        )
        compatibility_ok = compatibility_checks_ok and all(
            item.status == ExecutionStatus.APPLIED
            for item in compatibility_outcome.groups
        ) and not compatibility_outcome.terminal_error

        regular_config = dataclass_replace(
            config, optimization_groups=tuple(regular_entries)
        )
        preparation = Preparation(active_registry, handlers)
        if not compatibility_entries or compatibility_ok:
            regular_prepared = preparation.prepare(
                run_id=run_id,
                config=regular_config,
                environment=context,
                force_groups=tuple(
                    group_id for group_id in forced if group_id not in compatibility_ids
                ),
                import_missing=True,
            )
        else:
            blocked_groups = []
            for entry in regular_entries:
                definition = active_registry.get_group(entry.id)
                check_result = CheckResult(
                    "import_compatibility.blocked",
                    CheckStatus.FAIL,
                    expected="all import compatibility Groups applied",
                    actual="import compatibility application failed",
                )
                blocked_groups.append(
                    PreparedGroup(
                        entry.id,
                        definition.depends_on if definition else (),
                        definition.members if definition else (),
                        (check_result,),
                        Decision.BLOCK,
                        "import_compatibility_blocked",
                    )
                )
            regular_prepared = PreparedExecution(
                run_id,
                context,
                tuple(blocked_groups),
                (),
                (),
            )

        prepared_by_id = {
            group.group_id: group
            for group in (
                (compatibility_prepared.groups if compatibility_prepared else ())
                + regular_prepared.groups
            )
        }
        prepared_execution = PreparedExecution(
            run_id,
            context,
            tuple(
                prepared_by_id[entry.id]
                for entry in config.optimization_groups
                if entry.id in prepared_by_id
            ),
            (
                (compatibility_prepared.conflicts if compatibility_prepared else ())
                + regular_prepared.conflicts
            ),
            (
                (compatibility_prepared.execution_order if compatibility_prepared else ())
                + regular_prepared.execution_order
            ),
            regular_prepared.checks,
        )
    finally:
        if restore_imports:
            restore_results = Executor.restore_applied(
                compatibility_outcome.applied_snapshots
            )
            failed_restore = [
                item for item in restore_results if item.status.value == "failed"
            ]
            for name in set(sys.modules) - set(before_modules):
                sys.modules.pop(name, None)
            for name, module in before_modules.items():
                sys.modules[name] = module
            if failed_restore:
                raise OptimizationConfigError(
                    "failed to restore temporary import compatibility state: "
                    + "; ".join(item.error or item.path for item in failed_restore)
                )
    return (
        config,
        prepared_execution,
        preparation.prepared_groups,
        resolved_config_path,
        compatibility_outcome,
        regular_prepared,
    )


def check(
    *,
    optimization_config_path: Optional[PathLike] = None,
    registry: Optional[Registry] = None,
    catalog: Optional[OptimizationConfigCatalog] = None,
    force_groups: Sequence[str] = (),
) -> PreparedExecution:
    """Check applicability and resolve decisions without installing replacements.

    ``force_groups`` accepts only failed checks explicitly marked overrideable;
    structural errors remain blocked and are still included in the result.

    Target and fixed replacement resolution may import modules once.  A function
    declared through ``wrap()`` is called to construct its replacement, so wrapper
    construction must not perform irreversible side effects.
    The engine restores the ``sys.modules`` mapping on a best-effort basis, but it does
    not claim to undo arbitrary module import-time side effects.
    """

    _, prepared_execution, _, _, _, _ = _resolve(
        optimization_config_path=optimization_config_path,
        registry=registry,
        catalog=catalog,
        force_groups=force_groups,
        restore_imports=True,
    )
    return prepared_execution


def apply(
    *,
    optimization_config_path: Optional[PathLike] = None,
    report_dir: PathLike = "turbophysai_reports",
    registry: Optional[Registry] = None,
    catalog: Optional[OptimizationConfigCatalog] = None,
    force_groups: Sequence[str] = (),
) -> OptimizationReport:
    """Apply one OptimizationConfig during process startup and write its report.

    Call this entry point once per process, before importing the target model.
    Repeated application and in-process retry are intentionally unsupported.
    Blocked or successfully rolled-back Groups are isolated together with their
    dependents; unrelated Groups continue.  A rollback failure is terminal.
    ``force_groups`` bypasses only overrideable checks for the named enabled Groups.
    """

    _claim_apply()

    (
        config,
        prepared_execution,
        prepared_groups,
        resolved_config_path,
        compatibility_outcome,
        regular_prepared,
    ) = _resolve(
        optimization_config_path=optimization_config_path,
        registry=registry,
        catalog=catalog,
        force_groups=force_groups,
        restore_imports=False,
    )
    artifacts = (
        report_paths(Path(report_dir), prepared_execution.run_id)
        if prepared_execution.environment.rank == 0
        else ReportArtifacts()
    )
    regular_outcome: ExecutionOutcome = Executor().execute(
        regular_prepared, prepared_groups=prepared_groups
    )
    results_by_id = {
        item.group_id: item
        for item in compatibility_outcome.groups + regular_outcome.groups
    }
    execution = tuple(
        results_by_id[entry.id]
        for entry in config.optimization_groups
        if entry.id in results_by_id
    )
    report = write_report(
        build_report(
            config,
            prepared_execution,
            execution,
            optimization_config_path=str(resolved_config_path),
            runtime_config_path=os.environ.get(
                "TURBO_PHYSAI_RUNTIME_CONFIG_PATH"
            ),
            artifacts=artifacts,
        )
    )
    terminal_error = compatibility_outcome.terminal_error or regular_outcome.terminal_error
    if terminal_error:
        raise OptimizationRollbackError(terminal_error, report=report)
    return report


__all__ = [
    "Optimization",
    "apply",
    "check",
    "group",
    "import_alias",
    "optional_import",
    "registry_override",
    "replace",
    "replace_import",
    "wrap",
]
