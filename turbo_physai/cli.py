# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import difflib
import json
import os
import subprocess
import sys
from pathlib import Path

from .bootstrap import bootstrap_environment, isolation_flags
from .engine.errors import TurboPhysAIError
from .engine.contracts import to_primitive
from .engine.config.loader import (
    PACKAGED_DEFAULT_OPTIMIZATION_CONFIG,
    PACKAGED_OPTIMIZATION_ROOT,
    load_optimization_config,
)
from .engine.config.schema import optimization_config_to_dict
from .runtime import load_runtime_config, prepare_environment

_PACKAGED_MODEL_ROOT = PACKAGED_OPTIMIZATION_ROOT / "models"


def _run_training_command(command, environment) -> int:
    """Replace this process with the training command.

    ``execvpe`` keeps TurboPhysAI out of the process tree entirely, so the
    training process inherits the terminal, the job scheduler's process
    identity and its signal handling directly. Nothing needs forwarding.
    """

    if not command:
        raise TurboPhysAIError("a training command is required after --")
    try:
        os.execvpe(command[0], list(command), environment)
    except OSError as exc:
        raise TurboPhysAIError(
            f"failed to execute training command {command[0]!r}: {exc}"
        ) from exc
    raise AssertionError("unreachable: execvpe replaces this process")


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
        nargs="+",
        action="extend",
        default=[],
        metavar="GROUP_ID",
        help="force overrideable checks for one or more Groups in the current run",
    )
    run.add_argument(
        "--disable-group",
        nargs="+",
        action="extend",
        default=[],
        metavar="GROUP_ID",
        help="disable one or more Groups in the current run",
    )
    run.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    run.add_argument(
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
                numa_auto_override=False if args.disable_numa else None,
            )
            if runtime_path is None:
                environment.pop("TURBO_PHYSAI_RUNTIME_CONFIG_PATH", None)
            else:
                environment["TURBO_PHYSAI_RUNTIME_CONFIG_PATH"] = str(
                    runtime_path.resolve()
                )
            command = list(args.command)
            if command[:1] == ["--"]:
                command = command[1:]
            if not command:
                raise TurboPhysAIError("a training command is required after --")
            isolated = isolation_flags(command)
            if isolated:
                raise TurboPhysAIError(
                    f"cannot optimize a command using {' '.join(isolated)}: these "
                    "flags stop the interpreter from loading TurboPhysAI at "
                    "startup, which would silently run training unoptimized"
                )
            environment = bootstrap_environment(
                environment,
                optimization_config=str(optimization_config_path),
                report_dir=args.report_dir,
                force_groups=tuple(args.force_group),
                disable_groups=tuple(args.disable_group),
            )

            return _run_training_command(command, environment)
    except TurboPhysAIError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
