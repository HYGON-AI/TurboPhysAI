# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""RuntimeConfig loading and launch-time environment preparation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from .engine.errors import RuntimeConfigError


_CPU_RANGE = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
RUNTIME_CONFIG_SCHEMA_VERSION = "turbophysai/runtime-config/v1"


@dataclass(frozen=True)
class RuntimeConfig:
    environment_set: Mapping[str, str]
    environment_unset: tuple[str, ...]
    rank_affinity: Mapping[str, str]
    rank_numa: Mapping[str, int]
    numa_auto: bool


def parse_cpu_set(value: str) -> set[int]:
    """Parse a Linux CPU-list such as ``0-3,8,10-11``."""

    if not isinstance(value, str) or not _CPU_RANGE.fullmatch(value):
        raise RuntimeConfigError(
            f"invalid CPU affinity {value!r}; expected e.g. 0-3,8,10-11"
        )
    cpus: set[int] = set()
    for part in value.split(","):
        start, _, end = part.partition("-")
        first = int(start)
        last = int(end) if end else first
        if last < first:
            raise RuntimeConfigError(f"invalid descending CPU range: {part}")
        cpus.update(range(first, last + 1))
    return cpus


def parse_numa_node(value: str | int) -> int:
    """Parse a non-negative Linux NUMA node number."""

    try:
        node = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeConfigError(
            f"invalid NUMA node {value!r}; expected a non-negative integer"
        ) from exc
    if node < 0 or str(node) != str(value):
        raise RuntimeConfigError(
            f"invalid NUMA node {value!r}; expected a non-negative integer"
        )
    return node


def load_runtime_config(path: str | os.PathLike[str] | None) -> RuntimeConfig:
    if path is None:
        return RuntimeConfig({}, (), {}, {}, True)
    source = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeConfigError(f"failed to read RuntimeConfig {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeConfigError("RuntimeConfig must be a mapping")
    if (
        raw.get("schema_version") != RUNTIME_CONFIG_SCHEMA_VERSION
        or raw.get("kind") != "RuntimeConfig"
    ):
        raise RuntimeConfigError(
            "unsupported RuntimeConfig schema_version or kind"
        )
    environment = raw.get("environment", {})
    process = raw.get("process", {})
    if not isinstance(environment, dict) or not isinstance(process, dict):
        raise RuntimeConfigError("RuntimeConfig environment and process must be mappings")
    values = environment.get("set", {})
    unset = environment.get("unset", [])
    affinity = process.get("rank_affinity", {})
    numa = process.get("rank_numa", {})
    numa_auto = process.get("numa", "auto")
    if not isinstance(values, dict) or not all(isinstance(key, str) for key in values):
        raise RuntimeConfigError("environment.set must map variable names to values")
    if not isinstance(unset, list) or not all(isinstance(key, str) for key in unset):
        raise RuntimeConfigError("environment.unset must be a list of variable names")
    if not isinstance(affinity, dict):
        raise RuntimeConfigError("process.rank_affinity must be a mapping")
    if not isinstance(numa, dict):
        raise RuntimeConfigError("process.rank_numa must be a mapping")
    if not isinstance(numa_auto, bool) and numa_auto != "auto":
        raise RuntimeConfigError("process.numa must be false, true, or 'auto'")
    normalized_affinity = {str(rank): str(cpus) for rank, cpus in affinity.items()}
    for cpus in normalized_affinity.values():
        parse_cpu_set(cpus)
    normalized_numa = {str(rank): parse_numa_node(node) for rank, node in numa.items()}
    return RuntimeConfig(
        {key: str(value) for key, value in values.items()},
        tuple(unset),
        normalized_affinity,
        normalized_numa,
        numa_auto in {True, "auto"},
    )


def prepare_environment(
    runtime: RuntimeConfig,
    *,
    overrides: Mapping[str, str] = {},
    rank_affinity_overrides: Mapping[str, str] = {},
    rank_numa_overrides: Mapping[str, int] = {},
    numa_auto_override: bool | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for name in runtime.environment_unset:
        env.pop(name, None)
    env.update(runtime.environment_set)
    env.update(overrides)
    affinity = dict(runtime.rank_affinity)
    affinity.update(rank_affinity_overrides)
    for cpus in affinity.values():
        parse_cpu_set(cpus)
    if affinity:
        env["TURBO_PHYSAI_RANK_AFFINITY"] = json.dumps(affinity, sort_keys=True)
    numa = dict(runtime.rank_numa)
    numa.update(rank_numa_overrides)
    for node in numa.values():
        parse_numa_node(node)
    if numa_auto_override is False:
        numa.clear()
    if numa:
        env["TURBO_PHYSAI_RANK_NUMA"] = json.dumps(numa, sort_keys=True)
    else:
        env.pop("TURBO_PHYSAI_RANK_NUMA", None)
    numa_auto = runtime.numa_auto if numa_auto_override is None else numa_auto_override
    if numa_auto:
        env["TURBO_PHYSAI_NUMA_AUTO"] = "1"
    else:
        env.pop("TURBO_PHYSAI_NUMA_AUTO", None)
    return env
