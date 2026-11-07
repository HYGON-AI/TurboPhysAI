# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Per-launch Group option overrides, without rewriting generated configs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from ..contracts import OptimizationConfig
from ..errors import OptimizationConfigError


def parse_option_overrides(assignments: Sequence[str]) -> dict:
    result = {}
    for assignment in assignments:
        name, separator, raw = assignment.partition("=")
        group, dot, key = name.rpartition(".")
        if not separator or not group or not dot or not key or not raw:
            raise OptimizationConfigError(
                f"--option expects GROUP_ID.KEY=VALUE, got {assignment!r}"
            )
        if name in result:
            raise OptimizationConfigError(f"duplicate --option: {name}")
        try:
            result[name] = json.loads(raw)
        except json.JSONDecodeError:
            result[name] = raw
    return result


def apply_option_overrides(
    config: OptimizationConfig, overrides: Mapping | None
) -> OptimizationConfig:
    if overrides is None:
        return config
    if not isinstance(overrides, Mapping):
        raise OptimizationConfigError("option_overrides must be a mapping")
    entries = {entry.id: entry for entry in config.optimization_groups}
    for name, value in overrides.items():
        if not isinstance(name, str):
            raise OptimizationConfigError("option override keys must be strings")
        group, _, key = name.rpartition(".")
        entry = entries.get(group)
        if entry is None or key not in entry.options:
            raise OptimizationConfigError(f"unknown optimization option: {name}")
        previous = entry.options[key]
        if previous is not None:
            valid = (
                type(value) is type(previous)
                or isinstance(previous, tuple) and isinstance(value, (list, tuple))
                or isinstance(previous, Mapping) and isinstance(value, Mapping)
            )
            if not valid:
                raise OptimizationConfigError(
                    f"{name} expects {type(previous).__name__}, "
                    f"got {type(value).__name__}"
                )
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise OptimizationConfigError(f"{name} must be a JSON value") from exc
        entries[group] = replace(entry, options={**entry.options, key: value})
    return replace(
        config,
        optimization_groups=tuple(entries[entry.id] for entry in config.optimization_groups),
    )
