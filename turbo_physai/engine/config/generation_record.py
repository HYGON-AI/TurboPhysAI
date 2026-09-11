# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""File-level generation receipts; verification never imports model/Catalog code."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import yaml

from ..checking.evidence import _canonical_ast
from ..errors import OptimizationConfigError


SCHEMA = "turbophysai/generation-record/v2"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def record_path(config_path: Path) -> Path:
    return config_path.with_name("." + config_path.name + ".generation.json")


def _content_hash(content: bytes, role: str) -> str:
    # Parse only: Catalog code is never imported or executed here. Reuse the
    # engine's AST normalization to ignore positions and empty optional fields.
    if role == "catalog":
        normalized = _canonical_ast(ast.parse(content))
    else:
        # Preserve values (including whitespace inside strings) and sequence
        # order, while discarding YAML comments, layout and mapping key order.
        normalized = yaml.safe_dump(
            yaml.safe_load(content), sort_keys=True, allow_unicode=True,
        )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_hash(path: Path, role: str) -> str:
    try:
        return _content_hash(path.read_bytes(), role)
    except (SyntaxError, yaml.YAMLError, UnicodeError) as exc:
        raise OptimizationConfigError(f"cannot parse {role} file {path}: {exc}") from exc


def describe_input(path: Path, role: str, output: Path) -> dict:
    path = path.resolve()
    # Built-in dependencies remain portable across installs/checkouts. External
    # project files move together with their config using relative paths.
    try:
        relative = path.relative_to(PACKAGE_ROOT)
        base = "package"
    except ValueError:
        relative = Path(os.path.relpath(path, output.resolve().parent))
        base = "config"
    return {"role": role, "base": base, "path": relative.as_posix(), "sha256": file_hash(path, role)}


def _input_path(entry: dict, config_path: Path) -> Path:
    base = PACKAGE_ROOT if entry["base"] == "package" else config_path.resolve().parent
    return base / entry["path"]


def write_generated(config_path: Path, content: str, inputs: list, commit: str, *, force=False) -> None:
    receipt = record_path(config_path)
    if not force and (config_path.exists() or receipt.exists()):
        raise OptimizationConfigError(f"refusing to overwrite generated files: {config_path}; use --force")
    for entry in inputs:
        if file_hash(_input_path(entry, config_path), entry["role"]) != entry["sha256"]:
            raise OptimizationConfigError("generation inputs changed during generation; generate again")
    payload = content.encode("utf-8")
    record = {
        "schema_version": SCHEMA,
        "generator": "turbo-physai optimization generate",
        "model_commit": commit,
        "config": {"path": config_path.name, "sha256": _content_hash(payload, "config")},
        "inputs": inputs,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for destination, data in (
            (config_path, payload),
            (receipt, (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")),
        ):
            with tempfile.NamedTemporaryFile(dir=config_path.parent, delete=False) as stream:
                temporary.append((Path(stream.name), destination))
                stream.write(data)
        # Publish the receipt last. An interrupted pair is rejected by verification.
        for source, destination in temporary:
            os.replace(source, destination)
    finally:
        for source, _ in temporary:
            source.unlink(missing_ok=True)


def verify_generated(config_path: Path) -> dict:
    """Reject missing receipts and files changed since successful CLI generation."""
    config_path = Path(config_path)
    try:
        record = json.loads(record_path(config_path).read_text(encoding="utf-8"))
        if record["schema_version"] != SCHEMA or record["generator"] != "turbo-physai optimization generate":
            raise ValueError("unsupported generation record")
        if not isinstance(record["model_commit"], str) or not re.fullmatch(
            r"[0-9a-f]{40}", record["model_commit"]
        ):
            raise ValueError("invalid model commit in generation record")
        config = record["config"]
        inputs = record["inputs"]
        if config["path"] != config_path.name or not isinstance(inputs, list) or not inputs:
            raise ValueError("invalid generation record")
        if not any(entry["role"] == "recipe" for entry in inputs):
            raise ValueError("generation record has no recipe")
        for entry in inputs:
            if entry["role"] not in {"recipe", "catalog", "inherited_config"}:
                raise ValueError("invalid input role")
            if entry["base"] not in {"package", "config"} or Path(entry["path"]).is_absolute():
                raise ValueError("invalid input path")
        files = [(config_path, config["sha256"], "config")] + [
            (_input_path(entry, config_path), entry["sha256"], entry["role"]) for entry in inputs
        ]
        for path, expected, role in files:
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("invalid SHA-256 in generation record")
            if file_hash(path, role) != expected:
                raise ValueError(f"file changed since generation: {path}")
    except (OSError, ValueError, KeyError, TypeError, OptimizationConfigError) as exc:
        raise OptimizationConfigError(
            f"generation record verification failed for {config_path}: {exc}; "
            "regenerate with turbo-physai optimization generate"
        ) from exc
    return record
