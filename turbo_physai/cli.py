# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import difflib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .engine.errors import TurboPhysAIError
from .engine.contracts import to_primitive
from .engine.config.loader import (
    PACKAGED_DEFAULT_OPTIMIZATION_CONFIG,
    PACKAGED_OPTIMIZATION_ROOT,
    load_optimization_config,
)
from .engine.config.schema import optimization_config_to_dict
from .launchers import rewrite_command as _rewrite_command
from .runtime import load_runtime_config, parse_numa_node, prepare_environment

_INTERRUPT_GRACE_SECONDS = 30
_TERMINATE_GRACE_SECONDS = 5
_PACKAGED_MODEL_ROOT = PACKAGED_OPTIMIZATION_ROOT / "models"


class _LaunchSignal(Exception):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


def _signal_process_group(process, signum: int) -> None:
    """Signal the complete training process tree started by this CLI."""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def _wait_after_signal(process, signum: int) -> int:
    """Stop a training process group in stages and return a shell exit code."""

    _signal_process_group(process, signum)
    print(
        f"turbo-physai: forwarded {signal.Signals(signum).name} to training; "
        f"waiting up to {_INTERRUPT_GRACE_SECONDS}s",
        file=sys.stderr,
        flush=True,
    )
    try:
        process.wait(timeout=_INTERRUPT_GRACE_SECONDS)
    except (_LaunchSignal, subprocess.TimeoutExpired):
        _signal_process_group(process, signal.SIGTERM)
        print(
            "turbo-physai: training did not stop cleanly; forwarded SIGTERM",
            file=sys.stderr,
            flush=True,
        )
        try:
            process.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except (_LaunchSignal, subprocess.TimeoutExpired):
            _signal_process_group(process, signal.SIGKILL)
            print(
                "turbo-physai: training still running; forwarded SIGKILL",
                file=sys.stderr,
                flush=True,
            )
            process.wait()
    return 128 + signum


def _run_training_command(command, environment) -> int:
    """Launch training and reliably forward terminal stop signals."""

    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
    )
    previous_handlers = {}

    def receive_signal(signum, _frame):
        raise _LaunchSignal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, receive_signal)
    try:
        try:
            return process.wait()
        except _LaunchSignal as received:
            return _wait_after_signal(process, received.signum)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _json(value) -> str:
    return json.dumps(to_primitive(value), ensure_ascii=False, indent=2, sort_keys=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="turbo-physai")
    root = parser.add_subparsers(dest="resource", required=True)
    optimization = root.add_parser(
        "optimization", help="create and manage OptimizationConfig files"
    )
    optimization_commands = optimization.add_subparsers(
        dest="command", required=True
    )
    init = optimization_commands.add_parser("init")
    init.add_argument("name", help="model or optimization project name")
    init.add_argument(
        "--output",
        help="project directory (default: <name>_optimization)",
    )
    for command in ("validate", "show"):
        child = optimization_commands.add_parser(command)
        child.add_argument("optimization_config")
    check = optimization_commands.add_parser("check")
    check.add_argument(
        "optimization_config", help="generated OptimizationConfig YAML"
    )
    check.add_argument(
        "--repo",
        default=".",
        help="model repository to verify (default: current directory)",
    )
    diff = optimization_commands.add_parser("diff")
    diff.add_argument("left")
    diff.add_argument("right")
    generate = optimization_commands.add_parser("generate")
    generate.add_argument("--recipe", required=True)
    generate.add_argument("--repo", required=True)
    generate.add_argument("--commit", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--force", action="store_true")

    run = root.add_parser("run", help="prepare a RuntimeConfig and launch training")
    run.add_argument(
        "--model",
        help="built-in model name; automatically selects its optimization and runtime configs",
    )
    run.add_argument(
        "--optimization-config",
        help=(
            "explicit OptimizationConfig path; overrides the configuration "
            "selected by --model"
        ),
    )
    run.add_argument("--runtime-config", help="RuntimeConfig YAML")
    run.add_argument("--report-dir", default="turbophysai_reports")
    run.add_argument(
        "--force-group",
        action="append",
        default=[],
        metavar="GROUP_ID",
        help="force an overrideable check for this Group in the current run",
    )
    run.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument(
        "--set-rank-affinity", action="append", default=[], metavar="RANK=CPU_LIST"
    )
    run.add_argument(
        "--set-rank-numa", action="append", default=[], metavar="RANK=NODE",
        help="bind a rank to NUMA node with numactl",
    )
    numa_mode = run.add_mutually_exclusive_group()
    numa_mode.add_argument(
        "--enable-numa", action="store_true", help="auto-bind ranks by device topology"
    )
    numa_mode.add_argument(
        "--disable-numa", action="store_true", help="disable configured NUMA binding"
    )
    run.add_argument("command", nargs=argparse.REMAINDER, help="command after --")
    return parser


def _assignments(values, option):
    result = {}
    for value in values:
        name, separator, setting = value.partition("=")
        if not separator or not name or not setting:
            raise TurboPhysAIError(f"{option} expects NAME=VALUE, got {value!r}")
        result[name] = setting
    return result


def _numa_assignments(values):
    assignments = _assignments(values, "--set-rank-numa")
    return {rank: parse_numa_node(node) for rank, node in assignments.items()}


def _resolve_run_configs(model, optimization_config, runtime_config):
    """Resolve explicit, model-specific, or common built-in run configs."""

    model_root = None
    if model:
        model_name = model.strip().lower().replace("-", "_")
        if not model_name or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in model_name
        ):
            raise TurboPhysAIError(f"invalid model name: {model!r}")
        model_root = _PACKAGED_MODEL_ROOT / model_name / "configs"
        if not model_root.is_dir():
            available = sorted(
                path.name
                for path in _PACKAGED_MODEL_ROOT.iterdir()
                if (path / "configs" / "optimization.yaml").is_file()
            )
            choices = ", ".join(available) if available else "none"
            raise TurboPhysAIError(
                f"unknown built-in model {model!r}; available models: {choices}"
            )

    resolved_optimization = (
        Path(optimization_config).expanduser() if optimization_config else None
    )
    if resolved_optimization is None:
        resolved_optimization = (
            model_root / "optimization.yaml"
            if model_root is not None
            else PACKAGED_DEFAULT_OPTIMIZATION_CONFIG
        )
    if not resolved_optimization.is_file():
        raise TurboPhysAIError(
            f"OptimizationConfig not found: {resolved_optimization.resolve()}"
        )

    resolved_runtime = (
        Path(runtime_config).expanduser() if runtime_config else None
    )
    if resolved_runtime is None and model_root is not None:
        candidate = model_root / "runtime.yaml"
        if candidate.is_file():
            resolved_runtime = candidate
    if resolved_runtime is not None and not resolved_runtime.is_file():
        raise TurboPhysAIError(
            f"RuntimeConfig not found: {resolved_runtime.resolve()}"
        )
    return resolved_optimization.resolve(), (
        resolved_runtime.resolve() if resolved_runtime is not None else None
    )


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.resource == "optimization" and args.command in {"validate", "show"}:
            config = load_optimization_config(args.optimization_config)
            if args.command == "validate":
                print(
                    f"valid OptimizationConfig: "
                    f"{config.metadata.id} {config.metadata.version}"
                )
            else:
                print(_json(optimization_config_to_dict(config)))
            return 0
        if args.resource == "optimization" and args.command == "diff":
            left = _json(optimization_config_to_dict(load_optimization_config(args.left))).splitlines(
                True
            )
            right = _json(optimization_config_to_dict(load_optimization_config(args.right))).splitlines(
                True
            )
            sys.stdout.writelines(
                difflib.unified_diff(left, right, fromfile=args.left, tofile=args.right)
            )
            return 0
        if args.resource == "optimization" and args.command == "check":
            from .engine.config.generator import check_optimization_config

            config = check_optimization_config(
                Path(args.optimization_config), Path(args.repo)
            )
            print(
                f"checked OptimizationConfig: "
                f"{config.metadata.id} {config.metadata.version}"
            )
            return 0
        if args.resource == "optimization" and args.command == "generate":
            command = [
                sys.executable,
                "-m",
                "turbo_physai.engine.config.generator",
                "--recipe",
                args.recipe,
                "--repo",
                args.repo,
                "--commit",
                args.commit,
            ]
            completed = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode:
                print(completed.stderr.strip(), file=sys.stderr)
                return completed.returncode
            output = Path(args.output)
            if output.exists() and not args.force:
                print(f"refusing to overwrite existing file: {output}", file=sys.stderr)
                return 2
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(completed.stdout, encoding="utf-8")
            print(output)
            return 0
        if args.resource == "optimization" and args.command == "init":
            from .development import create_optimization_project

            output = Path(args.output) if args.output else None
            print(create_optimization_project(args.name, output))
            return 0
        if args.resource == "run":
            optimization_config_path, runtime_path = _resolve_run_configs(
                args.model, args.optimization_config, args.runtime_config
            )
            runtime = load_runtime_config(runtime_path)
            environment = prepare_environment(
                runtime,
                overrides=_assignments(args.set, "--set"),
                rank_affinity_overrides=_assignments(
                    args.set_rank_affinity, "--set-rank-affinity"
                ),
                rank_numa_overrides=_numa_assignments(args.set_rank_numa),
                numa_auto_override=(
                    True if args.enable_numa else False if args.disable_numa else None
                ),
            )
            if runtime_path is None:
                environment.pop("TURBO_PHYSAI_RUNTIME_CONFIG_PATH", None)
            else:
                environment["TURBO_PHYSAI_RUNTIME_CONFIG_PATH"] = str(
                    runtime_path.resolve()
                )
            command = _rewrite_command(
                args.command,
                optimization_config_path,
                args.report_dir,
                force_groups=tuple(args.force_group),
            )
            return _run_training_command(command, environment)
    except TurboPhysAIError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
