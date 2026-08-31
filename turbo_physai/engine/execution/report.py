# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import sys
from typing import Iterable, TextIO

from ..contracts import (
    Decision,
    ExecutionStatus,
    FrozenDict,
    OptimizationConfig,
    GroupExecutionResult,
    OptimizationReport,
    PreparedExecution,
    to_primitive,
)


def build_report(
    config: OptimizationConfig,
    prepared_execution: PreparedExecution,
    execution: Iterable[GroupExecutionResult],
    *,
    optimization_config_path: str,
    runtime_config_path: str | None = None,
) -> OptimizationReport:
    execution = tuple(execution)
    summary = {
        "applied": sum(item.status == ExecutionStatus.APPLIED for item in execution),
        "skipped": sum(
            item.decision == Decision.SKIP for item in prepared_execution.groups
        ),
        "blocked": sum(
            item.decision == Decision.BLOCK for item in prepared_execution.groups
        ),
        "failed": sum(item.status == ExecutionStatus.FAILED for item in execution),
        "rolled_back": sum(
            item.status == ExecutionStatus.ROLLED_BACK for item in execution
        ),
        "not_started": sum(
            item.status == ExecutionStatus.NOT_STARTED for item in execution
        ),
    }
    return OptimizationReport(
        run_id=prepared_execution.run_id,
        optimization_config=FrozenDict(to_primitive(config)),
        prepared_execution=prepared_execution,
        execution=execution,
        summary=FrozenDict(summary),
        optimization_config_path=optimization_config_path,
        runtime_config_path=runtime_config_path,
    )


def _log_value(value: object) -> str:
    return json.dumps(
        to_primitive(value), ensure_ascii=False, sort_keys=True, default=str
    )


def format_report(report: OptimizationReport) -> str:
    """Format an OptimizationReport for the training process log."""

    metadata = report.optimization_config["metadata"]
    lines = [
        f"TURBO_PHYSAI_OPTIMIZATION_REPORT_BEGIN run_id={report.run_id}",
        f"OptimizationConfig: {metadata['id']} {metadata['version']}",
        f"OptimizationConfig path: {report.optimization_config_path}",
        f"RuntimeConfig path: {report.runtime_config_path or 'not used'}",
        "Summary: "
        + " ".join(f"{name}={count}" for name, count in report.summary.items()),
        "Configuration checks:",
    ]
    if report.prepared_execution.checks:
        for check in report.prepared_execution.checks:
            lines.append(
                f"  - {check.code}: status={check.status.value} "
                f"expected={_log_value(check.expected)} "
                f"actual={_log_value(check.actual)}"
            )
    else:
        lines.append("  - none")

    lines.append("Preparation:")
    for group in report.prepared_execution.groups:
        dependencies = ",".join(group.depends_on) or "-"
        lines.append(
            f"  - {group.group_id}: dependencies={dependencies} "
            f"decision={group.decision.value} reason={group.reason}"
        )
        for check in group.checks:
            if check.status.value not in {"warning", "fail", "unknown"}:
                continue
            lines.append(
                f"    check {check.code}: status={check.status.value} "
                f"expected={_log_value(check.expected)} "
                f"actual={_log_value(check.actual)}"
            )

    lines.append("Execution:")
    for group in report.execution:
        line = f"  - {group.group_id}: status={group.status.value}"
        if group.error:
            line += f" error={_log_value(group.error)}"
        lines.append(line)
    lines.append(f"TURBO_PHYSAI_OPTIMIZATION_REPORT_END run_id={report.run_id}")
    return "\n".join(lines)


def emit_report(
    report: OptimizationReport, *, stream: TextIO | None = None
) -> OptimizationReport:
    """Write one complete OptimizationReport to the process log."""

    print(format_report(report), file=stream or sys.stdout, flush=True)
    return report
