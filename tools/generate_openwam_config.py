# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Generate OpenWAM target AST evidence without importing training dependencies.

This is a static configuration generator. It records identities of the clean
baseline's Python definitions; it does not certify runtime or numerical behavior.
Use the normal `turbo-physai optimization generate` in the training environment
when regenerating for a different baseline.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turbo_physai.engine.checking.evidence import _canonical_ast
from turbo_physai.engine.config.generation_record import describe_input, write_generated
from turbo_physai.engine.config.generator import (
    _expand_group_dependencies,
    _validate_group_composition,
)
from turbo_physai.engine.config.loader import resolve_optimization_config
from turbo_physai.engine.config.schema import (
    optimization_config_from_dict,
    optimization_config_to_dict,
)
from turbo_physai.engine.definitions.registry import default_registry

BASELINE = "1204dde1b9e9c232131fc99a3f2c6b78330e7f64"


def _git(repo, *args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
        encoding="utf-8",
    ).strip()


def definition(root, target):
    parts = target.split(".")
    for length in range(len(parts) - 1, 0, -1):
        path = root.joinpath(*parts[:length]).with_suffix(".py")
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        nodes = ast.parse(text).body
        node = None
        for part in parts[length:]:
            node = next(
                (
                    item
                    for item in nodes
                    if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                    and item.name == part
                ),
                None,
            )
            if node is None:
                raise ValueError(f"Cannot statically resolve {target}")
            nodes = node.body
        start = min([node.lineno] + [item.lineno for item in node.decorator_list])
        source = textwrap.dedent(
            "\n".join(text.splitlines()[start - 1 : node.end_lineno])
        )
        return (
            "ast-v1:"
            + hashlib.sha256(_canonical_ast(ast.parse(source)).encode()).hexdigest()
        )
    raise ValueError(f"Cannot resolve source file for {target}")


def generate(recipe, sources, output, *, force=False):
    raw = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    config = _expand_group_dependencies(
        resolve_optimization_config(optimization_config_from_dict(raw))
    )
    _validate_group_composition(config)
    result = optimization_config_to_dict(config)
    result["compatibility"] = {"commits": [BASELINE]}
    for entry in result["optimization_groups"]:
        hashes = {}
        group = default_registry.get_group(entry["id"])
        for member in group.members:
            spec = default_registry.get_spec(member)
            if spec.collect_target_hash:
                hashes[spec.target] = [definition(sources, spec.target)]
        entry["trust"] = {"ast_hashes": hashes}
    inputs = [describe_input(recipe, "recipe", output)]
    for name in config.optimization_modules:
        inputs.append(
            describe_input(
                Path(importlib.import_module(name).__file__), "catalog", output
            )
        )
    content = (
        "# Static target AST identities; no model execution was performed.\n"
        + yaml.safe_dump(result, sort_keys=False)
    )
    write_generated(output, content, inputs, BASELINE, force=force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / "turbo_physai/optimizations/models/openwam/configs/recipe.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "turbo_physai/optimizations/models/openwam/configs/optimization.yaml",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if _git(repo, "rev-parse", "HEAD") != BASELINE or _git(
        repo, "status", "--porcelain"
    ):
        parser.error(f"A clean OpenWAM checkout at {BASELINE} is required")
    generate(args.recipe.resolve(), repo, args.output.resolve(), force=args.force)


if __name__ == "__main__":
    main()
