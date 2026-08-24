# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from ..errors import ReportWriteError
from ..contracts import (
    Decision,
    ExecutionStatus,
    FrozenDict,
    OptimizationConfig,
    GroupExecutionResult,
    OptimizationReport,
    ReportArtifacts,
    PreparedExecution,
    to_primitive,
)


def report_paths(report_dir: Path, run_id: str) -> ReportArtifacts:
    root = Path(report_dir).expanduser().resolve()
    return ReportArtifacts(
        json_path=str(root / f"optimization_report-{run_id}.json"),
        markdown_path=str(root / f"optimization_report-{run_id}.md"),
    )


def build_report(
    config: OptimizationConfig,
    prepared_execution: PreparedExecution,
    execution: Iterable[GroupExecutionResult],
    *,
    optimization_config_path: str,
    runtime_config_path: str | None = None,
    artifacts: ReportArtifacts = ReportArtifacts(),
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
        artifacts=artifacts,
    )


def _markdown(report: OptimizationReport) -> str:
    lines = [
        "# TurboPhysAI Optimization Report",
        "",
        f"- Run ID: `{report.run_id}`",
        "- OptimizationConfig: "
        f"`{report.optimization_config['metadata']['id']}` "
        f"`{report.optimization_config['metadata']['version']}`",
        f"- OptimizationConfig path: `{report.optimization_config_path}`",
        "- RuntimeConfig path: "
        + (
            f"`{report.runtime_config_path}`"
            if report.runtime_config_path
            else "not used"
        ),
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|-|-:|",
    ]
    for name, count in report.summary.items():
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            "## Configuration Checks",
            "",
            "| Check | Status | Expected | Actual |",
            "|-|-|-|-|",
        ]
    )
    for check in report.prepared_execution.checks:
        lines.append(
            f"| `{check.code}` | {check.status.value} | "
            f"`{to_primitive(check.expected)}` | "
            f"`{to_primitive(check.actual)}` |"
        )
    lines.extend(
        [
            "",
            "## Preparation",
            "",
            "| Group | Dependencies | Decision | Reason |",
            "|-|-|-|-|",
        ]
    )
    for group in report.prepared_execution.groups:
        dependencies = ", ".join(f"`{item}`" for item in group.depends_on) or "-"
        lines.append(
            f"| `{group.group_id}` | {dependencies} | "
            f"{group.decision.value} | {group.reason} |"
        )
        notable = [
            check
            for check in group.checks
            if check.status.value in {"warning", "fail", "unknown"}
        ]
        for check in notable:
            lines.append(
                f"| ↳ `{check.code}` |  | {check.status.value} | "
                f"expected `{to_primitive(check.expected)}`, actual `{to_primitive(check.actual)}` |"
            )
    lines.extend(["", "## Execution", "", "| Group | Status | Error |", "|-|-|-|"])
    for group in report.execution:
        lines.append(
            f"| `{group.group_id}` | {group.status.value} | {group.error or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(report: OptimizationReport) -> OptimizationReport:
    if not report.artifacts.json_path or not report.artifacts.markdown_path:
        return report
    json_path = Path(report.artifacts.json_path)
    markdown_path = Path(report.artifacts.markdown_path)
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        # A shared run ID makes all rank-0 phases target the same report. Use
        # process-local temporary files so concurrent helpers cannot corrupt
        # each other's in-progress writes; the final replacements stay atomic.
        suffix = f".{os.getpid()}.tmp"
        json_tmp = json_path.with_name(json_path.name + suffix)
        markdown_tmp = markdown_path.with_name(markdown_path.name + suffix)
        with json_tmp.open("w", encoding="utf-8") as stream:
            json.dump(
                to_primitive(report),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
        with markdown_tmp.open("w", encoding="utf-8") as stream:
            stream.write(_markdown(report))
        json_tmp.replace(json_path)
        markdown_tmp.replace(markdown_path)
    except OSError as exc:
        raise ReportWriteError(
            f"failed to write OptimizationReport: {exc}", report=report, cause=exc
        ) from exc
    return report
