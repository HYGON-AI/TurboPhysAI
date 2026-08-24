# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch a Python training script after applying one TurboPhysAI OptimizationConfig.

The runner is deliberately model-agnostic: it installs the selected
replacements before executing the target script, then hands that script the
same command-line arguments it would receive from ``python <script>``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence

from . import apply
from .runtime import parse_cpu_set


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply a TurboPhysAI OptimizationConfig, then run a Python script."
    )
    parser.add_argument(
        "--optimization-config",
        default=os.environ.get("TURBO_PHYSAI_OPTIMIZATION_CONFIG"),
        help=(
            "OptimizationConfig path (defaults to "
            "TURBO_PHYSAI_OPTIMIZATION_CONFIG or the "
            "built-in configuration)"
        ),
    )
    parser.add_argument("--module", help="Run this module as __main__ instead of a script")
    parser.add_argument(
        "--report-dir",
        default=os.environ.get("TURBO_PHYSAI_REPORT_DIR", "turbophysai_reports"),
        help="OptimizationReport directory (default: turbophysai_reports)",
    )
    parser.add_argument(
        "--force-group",
        action="append",
        default=[],
        metavar="GROUP_ID",
        help="force an overrideable check for this Group in the current run",
    )
    parser.add_argument(
        "script_and_args",
        nargs=argparse.REMAINDER,
        metavar="-- SCRIPT [ARG ...]",
        help="Python script and arguments; put them after --",
    )
    return parser


def run(
    script: str,
    script_args: Sequence[str],
    *,
    optimization_config_path: str | None = None,
    report_dir: str = "turbophysai_reports",
    force_groups: Sequence[str] = (),
    apply_optimization: Callable[..., object] = apply,
) -> None:
    """Apply an OptimizationConfig once and execute ``script`` as ``__main__``."""

    target = Path(script).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"training script does not exist: {target}")

    _set_rank_numa_binding()
    _set_rank_affinity()

    report = apply_optimization(
        optimization_config_path=optimization_config_path,
        report_dir=report_dir,
        force_groups=tuple(force_groups),
    )
    _print_optimization_result(report)

    sys.argv = [str(target), *script_args]
    sys.path[0] = str(target.parent)
    runpy.run_path(str(target), run_name="__main__")


def _set_rank_affinity() -> None:
    raw = os.environ.get("TURBO_PHYSAI_RANK_AFFINITY")
    if not raw:
        return
    try:
        affinity = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("TURBO_PHYSAI_RANK_AFFINITY must be JSON") from exc
    local_rank = os.environ.get("LOCAL_RANK", "0")
    cpu_list = affinity.get(str(local_rank))
    if cpu_list is None:
        return
    requested = parse_cpu_set(cpu_list)
    allowed = os.sched_getaffinity(0)
    unavailable = requested - allowed
    if unavailable:
        raise RuntimeError(
            f"rank {local_rank} affinity requests unavailable CPUs: {sorted(unavailable)}"
        )
    os.sched_setaffinity(0, requested)


def _print_optimization_result(report: object) -> None:
    """Print the actual outcome without implying every Group applied."""

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    summary = getattr(report, "summary", {})
    fields = (
        "applied",
        "skipped",
        "blocked",
        "failed",
        "rolled_back",
        "not_started",
    )
    counts = " ".join(f"{name}={summary.get(name, 0)}" for name in fields)
    print(
        f"TURBO_PHYSAI_OPTIMIZATION_COMPLETED rank={rank} {counts} "
        f"run_id={report.run_id}",
        flush=True,
    )


def _set_rank_numa_binding() -> None:
    """Re-exec this rank through numactl before importing training code."""

    if os.environ.get("TURBO_PHYSAI_NUMA_BOUND") == "1":
        return
    local_rank = os.environ.get("LOCAL_RANK", "0")
    node = _configured_numa_node(local_rank)
    if node is None and os.environ.get("TURBO_PHYSAI_NUMA_AUTO") == "1":
        node = _discover_rank_numa_node(local_rank)
    if node is None:
        return
    if not isinstance(node, int) or node < 0:
        raise RuntimeError(
            f"rank {local_rank} NUMA node must be a non-negative integer"
        )
    numactl = shutil.which("numactl")
    if numactl is None:
        raise RuntimeError("rank NUMA binding requires numactl in PATH")
    environment = os.environ.copy()
    environment["TURBO_PHYSAI_NUMA_BOUND"] = "1"
    command = [
        numactl,
        f"--cpunodebind={node}",
        f"--membind={node}",
        sys.executable,
        "-m",
        "turbo_physai.runner",
        *sys.argv[1:],
    ]
    os.execvpe(numactl, command, environment)


def _configured_numa_node(local_rank: str) -> int | None:
    raw = os.environ.get("TURBO_PHYSAI_RANK_NUMA")
    if not raw:
        return None
    try:
        binding = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("TURBO_PHYSAI_RANK_NUMA must be JSON") from exc
    return binding.get(str(local_rank))


def _discover_rank_numa_node(local_rank: str) -> int:
    """Resolve ``LOCAL_RANK`` through HIP_VISIBLE_DEVICES and hy-smi topology."""

    try:
        rank = int(local_rank)
    except ValueError as exc:
        raise RuntimeError(f"LOCAL_RANK must be an integer, got {local_rank!r}") from exc
    visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    device = devices[rank] if devices and rank < len(devices) else str(rank)
    try:
        device_id = int(device)
    except ValueError as exc:
        raise RuntimeError(
            f"cannot infer NUMA node for HIP_VISIBLE_DEVICES entry {device!r}"
        ) from exc
    hy_smi = shutil.which("hy-smi")
    if hy_smi is None:
        raise RuntimeError("automatic NUMA binding requires hy-smi in PATH")
    completed = subprocess.run(
        [hy_smi, "--showtopo"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"hy-smi --showtopo failed: {completed.stderr.strip()}")
    pattern = rf"HCU\[{device_id}\]\s*:\s*\(Topology\) Numa Node\s+(\d+)"
    match = re.search(pattern, completed.stdout)
    if match is None:
        raise RuntimeError(
            f"hy-smi topology does not report a NUMA node for HCU[{device_id}]"
        )
    return int(match.group(1))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = args.script_and_args
    if command[:1] == ["--"]:
        command = command[1:]
    if args.module:
        _set_rank_numa_binding()
        _set_rank_affinity()
        report = apply(
            optimization_config_path=args.optimization_config,
            report_dir=args.report_dir,
            force_groups=tuple(args.force_group),
        )
        _print_optimization_result(report)
        sys.argv = [args.module, *command]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        return 0
    if not command:
        _parser().error("a Python script is required after --")
    run(
        command[0],
        command[1:],
        optimization_config_path=args.optimization_config,
        report_dir=args.report_dir,
        force_groups=tuple(args.force_group),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
