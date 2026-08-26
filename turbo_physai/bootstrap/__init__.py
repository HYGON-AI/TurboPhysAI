# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Activate TurboPhysAI inside a training rank without rewriting its command.

The launch layer injects ``PYTHONPATH`` and a few variables into the training
environment. ``_sitedir/sitecustomize.py`` is then imported by the standard
library ``site`` module at interpreter startup -- before the training script
runs any import -- and calls :func:`activate` from here.

This module must stay importable without Torch, HCU or any model dependency:
it runs in every descendant interpreter of a ``turbo-physai run``.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Mapping, Sequence

SITE_DIR = Path(__file__).resolve().parent / "_sitedir"

BOOTSTRAP_FLAG = "TURBO_PHYSAI_BOOTSTRAP"
OPTIMIZATION_CONFIG = "TURBO_PHYSAI_OPTIMIZATION_CONFIG"
RUNTIME_CONFIG_PATH = "TURBO_PHYSAI_RUNTIME_CONFIG_PATH"
RUN_ID = "TURBO_PHYSAI_RUN_ID"
LOG_REPORT = "TURBO_PHYSAI_LOG_REPORT"
FORCE_GROUPS = "TURBO_PHYSAI_FORCE_GROUPS"
DISABLE_GROUPS = "TURBO_PHYSAI_DISABLE_GROUPS"

# A rank that cannot install its OptimizationConfig must not fall through to an
# unoptimized run. site.execsitecustomize() swallows exceptions, so activation
# failures exit with this code instead of raising.
ACTIVATION_FAILURE_EXIT_CODE = 91

# Launcher processes are Python too, so they also import sitecustomize. They
# must not apply: they never import model code, and their children would then
# apply a second time. Unlike the torchrun option table this list tracks
# launcher names, which change far more slowly than launcher flags.
_LAUNCHER_COMMANDS = frozenset(
    {
        "torchrun",
        "torchpack",
        "deepspeed",
        "accelerate",
        "ray",
        "mpirun",
        "mpiexec",
        "srun",
        "horovodrun",
    }
)
_LAUNCHER_MODULES = frozenset(
    {
        "torch.distributed.run",
        "torch.distributed.launch",
        "deepspeed.launcher.runner",
        "accelerate.commands.launch",
        "torchpack.launch",
    }
)

# Command-line flags that disable the bootstrap channel: -E ignores PYTHONPATH,
# -I implies -E -s, and -S skips site entirely (so sitecustomize never runs).
_ISOLATION_FLAGS = frozenset({"E", "I", "S"})
_PYTHON_SHORT_FLAGS = frozenset("bBdEhiIOqsSuvVxX?")


def command_line() -> tuple[str, ...]:
    """Return this process' full argv, including interpreter flags.

    ``sys.argv`` drops the interpreter and its flags (``python -m pkg args``
    is seen as ``['-m', 'args']``), so launcher detection and NUMA re-exec both
    need the real command line.
    """

    try:
        raw = Path("/proc/self/cmdline").read_bytes()
    except OSError:
        return (sys.executable, *sys.argv)
    parts = raw.decode("utf-8", "surrogateescape").split("\0")
    if parts and parts[-1] == "":
        parts.pop()
    return tuple(parts)


def isolation_flags(command: Sequence[str]) -> tuple[str, ...]:
    """Return the ``-E``/``-I``/``-S`` flags present in ``command``.

    Only clusters made entirely of Python single-letter flags are considered,
    so a training argument such as ``--resume-from`` or ``-Iou`` is not
    mistaken for interpreter isolation.
    """

    found: list[str] = []
    for argument in command:
        if len(argument) < 2 or not argument.startswith("-") or argument[1] == "-":
            continue
        letters = argument[1:]
        if not set(letters) <= _PYTHON_SHORT_FLAGS:
            continue
        found.extend(f"-{letter}" for letter in letters if letter in _ISOLATION_FLAGS)
    return tuple(dict.fromkeys(found))


def _entry_is_launcher(command: Sequence[str]) -> bool:
    arguments = list(command[1:])
    while arguments:
        argument = arguments[0]
        if argument == "-m":
            module = arguments[1] if len(arguments) > 1 else ""
            return module in _LAUNCHER_MODULES
        if argument == "-c":
            return False
        if argument.startswith("-"):
            arguments.pop(0)
            continue
        name = Path(argument).name
        if name.endswith(".py"):
            name = name[: -len(".py")]
        return name in _LAUNCHER_COMMANDS
    return False


def should_activate(command: Sequence[str], environment: Mapping[str, str]) -> bool:
    """Report whether this interpreter is a training rank that must be patched."""

    if environment.get(BOOTSTRAP_FLAG) != "1":
        return False
    arguments = list(command[1:])
    while arguments and arguments[0].startswith("-") and arguments[0] not in {"-m", "-c"}:
        arguments.pop(0)
    if arguments[:1] == ["-c"]:
        # ``python -c`` is how multiprocessing spawns helpers; they inherit an
        # already-patched parent's environment but not its patched modules.
        return False
    if not arguments:
        # An interactive interpreter, not a training entry point.
        return False
    return not _entry_is_launcher(command)


def bootstrap_environment(
    environment: Mapping[str, str],
    *,
    optimization_config: str,
    log_report: bool = False,
    force_groups: Sequence[str] = (),
    disable_groups: Sequence[str] = (),
) -> dict[str, str]:
    """Add the variables that make a descendant interpreter self-activate."""

    prepared = dict(environment)
    existing = prepared.get("PYTHONPATH", "")
    entries = [str(SITE_DIR), *(entry for entry in existing.split(os.pathsep) if entry)]
    prepared["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    prepared[BOOTSTRAP_FLAG] = "1"
    prepared[OPTIMIZATION_CONFIG] = str(optimization_config)
    # One CLI launch can create several Python interpreters (launcher, ranks,
    # and model-side helper scripts). Give all descendants one identity so log
    # records produced by the same launch can be correlated.
    prepared[RUN_ID] = uuid.uuid4().hex
    if log_report:
        prepared[LOG_REPORT] = "1"
    else:
        prepared.pop(LOG_REPORT, None)
    if force_groups:
        prepared[FORCE_GROUPS] = os.pathsep.join(force_groups)
    else:
        prepared.pop(FORCE_GROUPS, None)
    if disable_groups:
        prepared[DISABLE_GROUPS] = os.pathsep.join(disable_groups)
    else:
        prepared.pop(DISABLE_GROUPS, None)
    return prepared


def _activate() -> None:
    # A script launched as ``python tools/train.py`` normally places
    # ``tools/`` rather than the repository root at ``sys.path[0]``. Model
    # repositories commonly expose project modules from that root, and
    # TurboPhysAI must resolve the same modules before training imports begin.
    working_directory = str(Path.cwd())
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or str(Path(entry).resolve()) != working_directory
    ]
    sys.path.insert(0, working_directory)

    from .. import apply
    from ..runner import (
        _print_optimization_result,
        _set_rank_affinity,
        _set_rank_numa_binding,
    )

    # Re-execs through numactl when this rank needs NUMA binding; the call does
    # not return in that case. The original command line is replayed verbatim.
    _set_rank_numa_binding(reexec_command=command_line())
    _set_rank_affinity()

    raw_groups = os.environ.get(FORCE_GROUPS, "")
    raw_disabled_groups = os.environ.get(DISABLE_GROUPS, "")
    report = apply(
        optimization_config_path=os.environ.get(OPTIMIZATION_CONFIG) or None,
        log_report=os.environ.get(LOG_REPORT) == "1",
        force_groups=tuple(group for group in raw_groups.split(os.pathsep) if group),
        disable_groups=tuple(
            group for group in raw_disabled_groups.split(os.pathsep) if group
        ),
    )
    if os.environ.get(LOG_REPORT) == "1":
        _print_optimization_result(report)


def activate() -> None:
    """Apply this rank's OptimizationConfig, or abort the rank."""

    if not should_activate(command_line(), os.environ):
        return
    try:
        _activate()
    except BaseException as error:  # noqa: BLE001 - see the comment below
        import traceback

        traceback.print_exc()
        print(
            f"turbo-physai: rank failed to apply its OptimizationConfig ({error}); "
            "aborting instead of training unoptimized",
            file=sys.stderr,
            flush=True,
        )
        # site.execsitecustomize() catches and merely prints exceptions raised
        # from sitecustomize, so raising here would let training continue
        # unoptimized and exit 0. os._exit bypasses that except clause.
        sys.stderr.flush()
        os._exit(ACTIVATION_FAILURE_EXIT_CODE)
