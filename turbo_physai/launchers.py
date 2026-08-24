# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Rewrite supported training launchers so every Python rank enters the Runner."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from .engine.errors import TurboPhysAIError


_PYTHON_FLAG_OPTIONS = {
    "-B",
    "-E",
    "-I",
    "-O",
    "-OO",
    "-P",
    "-S",
    "-b",
    "-bb",
    "-q",
    "-s",
    "-u",
    "-v",
}
_PYTHON_VALUE_OPTIONS = {"-W", "-X"}
_TORCHRUN_VALUE_OPTIONS = {
    "--nnodes",
    "--nproc-per-node",
    "--nproc_per_node",
    "--node-rank",
    "--node_rank",
    "--master-addr",
    "--master_addr",
    "--master-port",
    "--master_port",
    "--rdzv-backend",
    "--rdzv_backend",
    "--rdzv-endpoint",
    "--rdzv_endpoint",
    "--rdzv-id",
    "--rdzv_id",
    "--max-restarts",
    "--max_restarts",
    "--monitor-interval",
    "--monitor_interval",
    "--role",
    "--logs-specs",
    "--logs_specs",
    "--local-ranks-filter",
    "--local_ranks_filter",
}


def _runner_arguments(
    optimization_config: str,
    report_dir: str,
    entry: str,
    train_args: Sequence[str],
    *,
    module: bool = False,
    force_groups: Sequence[str] = (),
) -> list[str]:
    command = [
        "-m",
        "turbo_physai.runner",
        "--optimization-config",
        str(Path(optimization_config).resolve()),
        "--report-dir",
        str(Path(report_dir).resolve()),
    ]
    for group_id in force_groups:
        command.extend(["--force-group", group_id])
    if module:
        return command + ["--module", entry, "--", *train_args]
    return command + ["--", entry, *train_args]


def _is_python_executable(value: str) -> bool:
    name = Path(value).name
    return bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name))


def _parse_python_command(command: Sequence[str]):
    """Split ``python [options] entry [args]`` without executing it."""

    if not command or not _is_python_executable(command[0]):
        raise TurboPhysAIError("a supported Python interpreter is required")
    interpreter = command[0]
    options: list[str] = []
    index = 1
    while index < len(command):
        item = command[index]
        if item == "--":
            index += 1
            break
        if item == "-m":
            if index + 1 >= len(command):
                raise TurboPhysAIError("python -m requires a module")
            return interpreter, options, command[index + 1], command[index + 2 :], True
        if item == "-c":
            raise TurboPhysAIError("python -c is not supported by turbo-physai run")
        if item in _PYTHON_FLAG_OPTIONS:
            options.append(item)
            index += 1
            continue
        if item in _PYTHON_VALUE_OPTIONS:
            if index + 1 >= len(command):
                raise TurboPhysAIError(f"{item} requires a value")
            options.extend((item, command[index + 1]))
            index += 2
            continue
        if any(
            item.startswith(option) and item != option
            for option in _PYTHON_VALUE_OPTIONS
        ):
            options.append(item)
            index += 1
            continue
        if item.startswith("-"):
            raise TurboPhysAIError(
                f"unsupported Python launcher option: {item}"
            )
        break
    if index >= len(command):
        raise TurboPhysAIError("Python command is missing its training entry")
    return interpreter, options, command[index], command[index + 1 :], False


class LauncherAdapter(ABC):
    """A bounded command grammar that can locate one Python training entry."""

    @abstractmethod
    def matches(self, command: Sequence[str]) -> bool:
        raise NotImplementedError

    @abstractmethod
    def rewrite(
        self,
        command: Sequence[str],
        optimization_config: str,
        report_dir: str,
        force_groups: Sequence[str],
    ) -> list[str]:
        raise NotImplementedError


class PythonAdapter(LauncherAdapter):
    def matches(self, command: Sequence[str]) -> bool:
        return bool(command) and _is_python_executable(command[0])

    def rewrite(
        self,
        command: Sequence[str],
        optimization_config: str,
        report_dir: str,
        force_groups: Sequence[str],
    ) -> list[str]:
        interpreter, options, entry, train_args, module = _parse_python_command(
            command
        )
        return [
            interpreter,
            *options,
            *_runner_arguments(
                optimization_config,
                report_dir,
                entry,
                train_args,
                module=module,
                force_groups=force_groups,
            ),
        ]


class TorchrunAdapter(LauncherAdapter):
    def matches(self, command: Sequence[str]) -> bool:
        return bool(command) and Path(command[0]).name == "torchrun"

    def rewrite(
        self,
        command: Sequence[str],
        optimization_config: str,
        report_dir: str,
        force_groups: Sequence[str],
    ) -> list[str]:
        index = 1
        while index < len(command):
            item = command[index]
            if item == "--no-python":
                raise TurboPhysAIError(
                    "torchrun --no-python is not supported by turbo-physai run"
                )
            if item == "-m" or not item.startswith("-"):
                break
            index += 1
            if (
                item.split("=", 1)[0] in _TORCHRUN_VALUE_OPTIONS
                and "=" not in item
            ):
                index += 1
        if index >= len(command):
            raise TurboPhysAIError("torchrun command is missing its training entry")
        module = command[index] == "-m"
        if module and index + 1 >= len(command):
            raise TurboPhysAIError("torchrun -m requires a module")
        entry_index = index + 1 if module else index
        return [
            *command[:index],
            *_runner_arguments(
                optimization_config,
                report_dir,
                command[entry_index],
                command[entry_index + 1 :],
                module=module,
                force_groups=force_groups,
            ),
        ]


class TorchPackAdapter(LauncherAdapter):
    def matches(self, command: Sequence[str]) -> bool:
        return len(command) >= 2 and list(command[:2]) == [
            "torchpack",
            "dist-run",
        ]

    def rewrite(
        self,
        command: Sequence[str],
        optimization_config: str,
        report_dir: str,
        force_groups: Sequence[str],
    ) -> list[str]:
        python_index = next(
            (
                index
                for index in range(2, len(command))
                if _is_python_executable(command[index])
            ),
            None,
        )
        if python_index is None:
            raise TurboPhysAIError(
                "torchpack dist-run requires a supported Python command"
            )
        interpreter, options, entry, train_args, module = _parse_python_command(
            command[python_index:]
        )
        return [
            *command[:python_index],
            interpreter,
            *options,
            *_runner_arguments(
                optimization_config,
                report_dir,
                entry,
                train_args,
                module=module,
                force_groups=force_groups,
            ),
        ]


_ADAPTERS = (PythonAdapter(), TorchrunAdapter(), TorchPackAdapter())


def rewrite_command(
    command: Sequence[str],
    optimization_config: str,
    report_dir: str,
    force_groups: Sequence[str] = (),
) -> list[str]:
    if command[:1] == ["--"]:
        command = command[1:]
    for adapter in _ADAPTERS:
        if adapter.matches(command):
            return adapter.rewrite(
                command,
                optimization_config,
                report_dir,
                force_groups,
            )
    raise TurboPhysAIError(
        "run supports only Python, torchrun, and torchpack dist-run commands"
    )


__all__ = [
    "LauncherAdapter",
    "PythonAdapter",
    "TorchPackAdapter",
    "TorchrunAdapter",
    "rewrite_command",
]
