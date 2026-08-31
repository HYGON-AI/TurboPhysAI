# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.metadata
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

from ..contracts import CompatibilityTarget, EnvironmentSnapshot, FrozenDict


def _run_git(args, cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git"] + list(args),
            cwd=str(cwd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def detect_context(
    *,
    dependency_names: Iterable[str] = (),
    project_dir: Optional[Path] = None,
    backend: Optional[str] = None,
) -> EnvironmentSnapshot:
    project_dir = (project_dir or Path.cwd()).resolve()
    dependencies = {}
    for name in sorted(set(dependency_names)):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None

    repository_root = _run_git(["rev-parse", "--show-toplevel"], project_dir)
    git_cwd = Path(repository_root) if repository_root else project_dir
    commit = _run_git(["rev-parse", "HEAD"], git_cwd) if repository_root else None
    status = _run_git(["status", "--porcelain"], git_cwd) if repository_root else None
    remote = (
        _run_git(["config", "--get", "remote.origin.url"], git_cwd)
        if repository_root
        else None
    )

    return EnvironmentSnapshot(
        python_version=platform.python_version(),
        platform=platform.platform(),
        executable=sys.executable,
        cwd=str(project_dir),
        dependencies=FrozenDict(dependencies),
        repository=remote or repository_root,
        commit=commit,
        dirty=None if status is None else bool(status),
        backend=backend or os.environ.get("TURBO_PHYSAI_BACKEND"),
        rank=_int_env("RANK", 0),
        local_rank=_int_env("LOCAL_RANK", 0),
        world_size=_int_env("WORLD_SIZE", 1),
    )


def detect_target_context(path: str, value: object) -> CompatibilityTarget:
    """Describe the source checkout that owns one resolved target, if present."""

    try:
        source = inspect.getsourcefile(inspect.unwrap(value))
    except (TypeError, ValueError):
        source = None
    if not source:
        return CompatibilityTarget(path)
    source_path = Path(source).resolve()
    if not source_path.exists():
        return CompatibilityTarget(path, source_file=str(source_path))
    repository_root = _run_git(
        ["rev-parse", "--show-toplevel"], source_path.parent
    )
    if not repository_root:
        return CompatibilityTarget(path, source_file=str(source_path))
    root = Path(repository_root)
    commit = _run_git(["rev-parse", "HEAD"], root)
    status = _run_git(["status", "--porcelain"], root)
    remote = _run_git(["config", "--get", "remote.origin.url"], root)
    return CompatibilityTarget(
        path=path,
        source_file=str(source_path),
        repository_root=str(root),
        repository=remote or str(root),
        commit=commit,
        dirty=None if status is None else bool(status),
    )
