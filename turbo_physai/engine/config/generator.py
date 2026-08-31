# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate and verify OptimizationConfig evidence against a model checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml

from ..checking.evidence import ast_hash, source_hash
from .reference_extraction import direct_reference_lines
from ..checking.conflicts import find_group_conflicts, format_conflict
from ..errors import OptimizationConfigError
from ..contracts import (
    Mechanism,
    OptimizationConfig,
    OPTIMIZATION_CONFIG_SCHEMA_VERSION,
    OptimizationGroupConfig,
)
from ..definitions.registry import default_registry
from ..execution.replacements.base import (
    HandlerError,
    resolve_attribute,
)
from ..execution.replacements import default_handlers
from .loader import OptimizationConfigCatalog, load_optimization_config, resolve_optimization_config
from .schema import optimization_config_from_dict, optimization_config_to_dict


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise OptimizationConfigError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def _validate_group_composition(config) -> None:
    enabled_ids = [entry.id for entry in config.optimization_groups if entry.enabled]
    missing = [
        group_id
        for group_id in enabled_ids
        if default_registry.get_group(group_id) is None
    ]
    if missing:
        raise OptimizationConfigError(
            "OptimizationGroup is not registered: " + ", ".join(missing)
        )
    conflicts = find_group_conflicts(default_registry, enabled_ids)
    if conflicts:
        evidence = "\n".join(
            f"- {format_conflict(conflict)}" for conflict in conflicts
        )
        raise OptimizationConfigError(
            "OptimizationConfig has Group conflicts:\n" + evidence
        )


def _inherited_group_ids(config: OptimizationConfig, catalog: OptimizationConfigCatalog) -> Tuple[str, ...]:
    """Return Group IDs contributed by Configs named in ``extends``."""

    inherited = OptimizationConfig(
        schema_version=config.schema_version,
        kind=config.kind,
        metadata=config.metadata,
        extends=config.extends,
    )
    resolved = resolve_optimization_config(inherited, catalog=catalog)
    resolved = _expand_group_dependencies(resolved)
    return tuple(entry.id for entry in resolved.optimization_groups)


def _validate_public_replacement_references(
    config: OptimizationConfig,
    inherited_group_ids: Iterable[str],
    repo: Path,
) -> None:
    """Reject model code that bypasses an inherited public target entry."""

    inherited_ids = set(inherited_group_ids)
    public_replacements: Dict[str, list[Tuple[str, str, str]]] = {}
    for group_id in inherited_ids:
        group = default_registry.get_group(group_id)
        if group is None:
            continue
        for replacement_id in group.members:
            spec = default_registry.get_spec(replacement_id)
            if spec is not None:
                public_replacements.setdefault(spec.replacement, []).append(
                    (group_id, replacement_id, spec.target)
                )
    if not public_replacements:
        return

    failures = []
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    try:
        for entry in config.optimization_groups:
            if not entry.enabled or entry.id in inherited_ids:
                continue
            group = default_registry.get_group(entry.id)
            if group is None:
                continue
            for replacement_id in group.members:
                spec = default_registry.get_spec(replacement_id)
                if spec is None:
                    continue
                for public_path, line in direct_reference_lines(
                    spec.replacement, public_replacements
                ):
                    for public_group, public_replacement, public_target in (
                        public_replacements[public_path]
                    ):
                        failures.append(
                            "code=public.replacement_reference, "
                            f"model_group={entry.id}, model_replacement={replacement_id}, "
                            f"model_replacement={spec.replacement}, line={line}, "
                            f"public_group={public_group}, "
                            f"public_replacement={public_replacement}, "
                            f"public_replacement={public_path}, "
                            f"use_standard_target={public_target}"
                        )
    finally:
        sys.path.remove(str(repo))
    if failures:
        raise OptimizationConfigError(
            "Model Replacement directly references an inherited public "
            "Replacement. Call the standard target so the public Group remains "
            "independently selectable:\n- " + "\n- ".join(failures)
        )


def _expand_group_dependencies(config):
    """Materialize the dependency closure using Catalog declarations."""

    entries = {entry.id: entry for entry in config.optimization_groups}
    ordered = []
    visited = set()
    visiting = []

    def visit(group_id: str) -> None:
        if group_id in visited:
            return
        if group_id in visiting:
            cycle = visiting[visiting.index(group_id) :] + [group_id]
            raise OptimizationConfigError(
                "OptimizationGroup dependency cycle: " + " -> ".join(cycle)
            )
        group = default_registry.get_group(group_id)
        if group is None:
            raise OptimizationConfigError(f"OptimizationGroup is not registered: {group_id}")

        visiting.append(group_id)
        for dependency in group.depends_on:
            dependency_entry = entries.get(dependency)
            if dependency_entry is not None and not dependency_entry.enabled:
                raise OptimizationConfigError(
                    f"OptimizationGroup {group_id} depends on disabled Group {dependency}"
                )
            if dependency_entry is None:
                entries[dependency] = OptimizationGroupConfig(dependency)
            visit(dependency)
        visiting.pop()
        visited.add(group_id)
        ordered.append(entries[group_id])

    for entry in config.optimization_groups:
        if entry.enabled:
            visit(entry.id)
    ordered.extend(entry for entry in config.optimization_groups if not entry.enabled)
    return replace(config, optimization_groups=tuple(ordered))


def _validate_repository(
    repo: Path,
    *,
    expected_commit: str | None = None,
) -> None:
    repo = repo.resolve()
    if expected_commit is not None:
        head = _git(repo, "rev-parse", "HEAD")
        if head != expected_commit:
            raise OptimizationConfigError(
                "repository HEAD mismatch: "
                f"expected {expected_commit}, actual {head}"
            )
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise OptimizationConfigError(
            "OptimizationConfig generation requires a clean model worktree"
        )


def _collect_group_evidence(
    config: OptimizationConfig,
    repo: Path,
) -> Dict[str, Dict[str, Dict[str, list]]]:
    evidence = {}
    repo = repo.resolve()
    sys.path.insert(0, str(repo))
    previous = Path.cwd()
    try:
        os.chdir(repo)
        with _temporary_import_compatibility(config):
            for entry in config.optimization_groups:
                if not entry.enabled:
                    continue
                group = default_registry.get_group(entry.id)
                if group is None:
                    raise OptimizationConfigError(
                        f"OptimizationGroup is not registered: {entry.id}"
                    )
                source_hashes = {}
                ast_hashes = {}
                for replacement_id in group.members:
                    spec = default_registry.get_spec(replacement_id)
                    if spec is None:
                        raise OptimizationConfigError(
                            f"ReplacementSpec is not registered: {replacement_id}"
                        )
                    # Import compatibility and complete module replacement do not
                    # identify an importable upstream callable for source evidence.
                    if spec.mechanism in {
                        Mechanism.IMPORT_REPLACE,
                        Mechanism.IMPORT_ALIAS,
                        Mechanism.OPTIONAL_IMPORT,
                        Mechanism.REGISTRY_OVERRIDE,
                    }:
                        continue
                    try:
                        original = resolve_attribute(spec.target).original
                    except HandlerError as exc:
                        raise OptimizationConfigError(
                            f"cannot resolve evidence object {spec.target}: {exc}"
                        ) from exc
                    source = source_hash(original)
                    syntax = ast_hash(original)
                    if source is not None:
                        source_hashes[spec.target] = [source]
                    if syntax is not None:
                        ast_hashes[spec.target] = [syntax]
                evidence[entry.id] = {
                    "source_hashes": source_hashes,
                    "ast_hashes": ast_hashes,
                }
    finally:
        os.chdir(previous)
        sys.path.remove(str(repo))
    return evidence


@contextmanager
def _temporary_import_compatibility(config: OptimizationConfig):
    """Apply selected import compatibility Groups while collecting evidence."""

    mechanisms = {
        Mechanism.IMPORT_REPLACE,
        Mechanism.IMPORT_ALIAS,
        Mechanism.OPTIONAL_IMPORT,
        Mechanism.REGISTRY_OVERRIDE,
    }
    handlers = default_handlers()
    snapshots = []
    try:
        for entry in config.optimization_groups:
            if not entry.enabled:
                continue
            group = default_registry.get_group(entry.id)
            if group is None:
                continue
            specs = tuple(default_registry.get_spec(item) for item in group.members)
            if not specs or not any(
                spec is not None and spec.mechanism in mechanisms for spec in specs
            ):
                continue
            if any(spec is None or spec.mechanism not in mechanisms for spec in specs):
                raise OptimizationConfigError(
                    "import compatibility and runtime replacements must use "
                    f"separate OptimizationGroups: {entry.id}"
                )
            prepared = []
            try:
                for spec in specs:
                    handler = handlers[spec.mechanism]
                    item = handler.prepare(spec, entry.options)
                    prepared.append((handler, item, handler.snapshot(item)))
                for handler, item, _ in prepared:
                    handler.apply(item)
                snapshots.extend((handler, snapshot) for handler, _, snapshot in prepared)
            except Exception as exc:
                for handler, _, snapshot in reversed(prepared):
                    handler.restore(snapshot)
                raise OptimizationConfigError(
                    f"failed to apply import compatibility Group {entry.id}: {exc}"
                ) from exc
        yield
    finally:
        failures = []
        for handler, snapshot in reversed(snapshots):
            failures.extend(
                result
                for result in handler.restore(snapshot)
                if result.status.value == "failed"
            )
        if failures:
            raise OptimizationConfigError(
                "failed to restore import compatibility after evidence collection: "
                + "; ".join(result.error or result.path for result in failures)
            )


def _accepted_hashes(value) -> tuple:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _validate_evidence(
    config: OptimizationConfig,
    actual_by_group: Mapping[str, Mapping[str, Mapping[str, Sequence[str]]]],
) -> None:
    failures = []
    for entry in config.optimization_groups:
        if not entry.enabled:
            continue
        actual = actual_by_group.get(entry.id, {})
        for category in ("source_hashes", "ast_hashes"):
            expected_hashes = entry.trust.get(category, {})
            if not isinstance(expected_hashes, Mapping):
                expected_hashes = {}
            actual_hashes = actual.get(category, {})
            expected_targets = set(expected_hashes)
            actual_targets = set(actual_hashes)
            for target in sorted(expected_targets - actual_targets):
                failures.append(
                    f"group={entry.id}, target={target}, {category} "
                    "is no longer available"
                )
            for target in sorted(actual_targets):
                actual_values = tuple(actual_hashes[target])
                accepted = _accepted_hashes(expected_hashes.get(target, ()))
                if not actual_values or any(
                    value not in accepted for value in actual_values
                ):
                    failures.append(
                        f"group={entry.id}, target={target}, {category} "
                        f"expected={accepted}, actual={actual_values}"
                    )
    if failures:
        raise OptimizationConfigError(
            "OptimizationConfig target evidence mismatch:\n- "
            + "\n- ".join(failures)
        )


def generate(recipe: Path, repo: Path, commit: str) -> str:
    repo = repo.resolve()
    _validate_repository(repo, expected_commit=commit)

    recipe_raw = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    if not isinstance(recipe_raw, Mapping):
        raise OptimizationConfigError("OptimizationConfig recipe must be a mapping")
    recipe_raw = dict(recipe_raw)
    # Configuration identity is owned by the framework, not by optimization
    # developers. Always stamp the current schema when producing the artifact.
    recipe_raw["schema_version"] = OPTIMIZATION_CONFIG_SCHEMA_VERSION
    recipe_raw["kind"] = "OptimizationConfig"
    model = recipe_raw.get("model", {})
    if isinstance(model, Mapping) and model.get("name"):
        recipe_raw.setdefault("compatibility", {})["commits"] = [commit]
    recipe_config = optimization_config_from_dict(recipe_raw)
    catalog = OptimizationConfigCatalog.from_builtin_files()
    inherited_group_ids = _inherited_group_ids(recipe_config, catalog)
    config = resolve_optimization_config(recipe_config, catalog=catalog)
    config = _expand_group_dependencies(config)
    _validate_group_composition(config)
    _validate_public_replacement_references(config, inherited_group_ids, repo)

    # Flatten inherited and dependent Group selection into the generated file.
    # Target/replacement/dependency declarations remain owned by the Catalog.
    raw = optimization_config_to_dict(config)
    raw.pop("extends", None)
    evidence = _collect_group_evidence(config, repo)
    for entry in raw.get("optimization_groups", []):
        if entry.get("enabled", True):
            entry["trust"] = evidence[entry["id"]]

    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)


def check_optimization_config(
    optimization_config_path: Path, repo: Path
) -> OptimizationConfig:
    """Validate one generated OptimizationConfig against a clean checkout."""

    config = load_optimization_config(optimization_config_path)
    repo = repo.resolve()
    _validate_repository(repo)
    expanded = _expand_group_dependencies(config)
    actual_order = tuple(entry.id for entry in config.optimization_groups)
    generated_order = tuple(entry.id for entry in expanded.optimization_groups)
    if actual_order != generated_order:
        raise OptimizationConfigError(
            "OptimizationConfig Group dependency closure/order mismatch: "
            f"expected {generated_order}, actual {actual_order}"
        )
    _validate_group_composition(config)
    _validate_evidence(config, _collect_group_evidence(config, repo))
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    try:
        sys.stdout.write(
            generate(Path(args.recipe), Path(args.repo), args.commit)
        )
    except OptimizationConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
