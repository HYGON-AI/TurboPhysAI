# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from ..errors import OptimizationConfigError
from ..contracts import (
    FrozenDict,
    OptimizationConfig,
    OPTIMIZATION_CONFIG_SCHEMA_VERSION,
    OptimizationGroupConfig,
    OptimizationConfigMetadata,
    to_primitive,
)


_TOP_LEVEL = {
    "schema_version",
    "kind",
    "metadata",
    "model",
    "optimization_modules",
    "extends",
    "compatibility",
    "optimization_groups",
}
_METADATA = {"id", "version", "description"}
_MODEL = {"name"}
_COMPATIBILITY = {"commits", "allow_dirty", "dependencies", "backend", "repository"}
_GROUP = {"id", "enabled", "options", "trust"}
_TRUST = {"ast_hashes", "source_hashes"}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OptimizationConfigError(f"{path} must be a mapping")
    return value


def _reject_unknown(
    value: Mapping[str, Any], allowed: Iterable[str], path: str
) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise OptimizationConfigError(f"unknown fields at {path}: {', '.join(unknown)}")


def _json_only(value: Any, path: str) -> Any:
    if callable(value):
        raise OptimizationConfigError(f"Python callable is not allowed at {path}")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_only(item, f"{path}.{key}") for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_only(item, f"{path}[]") for item in value]
    raise OptimizationConfigError(f"unsupported value at {path}: {type(value).__name__}")


def _string_or_list(value: Any, path: str) -> None:
    if isinstance(value, str):
        return
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return
    raise OptimizationConfigError(f"{path} must be a string or list of strings")


def _validate_compatibility(value: Mapping[str, Any]) -> None:
    if "allow_dirty" in value and not isinstance(value["allow_dirty"], bool):
        raise OptimizationConfigError("compatibility.allow_dirty must be boolean")
    for field in ("commits", "backend", "repository"):
        if field in value:
            _string_or_list(value[field], f"compatibility.{field}")
    dependencies = value.get("dependencies", {})
    if not isinstance(dependencies, Mapping):
        raise OptimizationConfigError("compatibility.dependencies must be a mapping")
    for name, accepted in dependencies.items():
        _string_or_list(accepted, f"compatibility.dependencies.{name}")
        values = (accepted,) if isinstance(accepted, str) else tuple(accepted)
        for requirement in values:
            if requirement.startswith(("<", ">", "=", "!", "~")):
                try:
                    SpecifierSet(requirement)
                except InvalidSpecifier as exc:
                    raise OptimizationConfigError(
                        f"invalid version range at compatibility.dependencies.{name}: "
                        f"{requirement}"
                    ) from exc


def _validate_trust(value: Mapping[str, Any], path: str) -> None:
    for category in ("ast_hashes", "source_hashes"):
        hashes = value.get(category, {})
        if not isinstance(hashes, Mapping):
            raise OptimizationConfigError(f"{path}.{category} must be a mapping")
        for target, accepted in hashes.items():
            _string_or_list(accepted, f"{path}.{category}.{target}")


def optimization_config_from_dict(raw: Mapping[str, Any]) -> OptimizationConfig:
    raw = _mapping(raw, "config")
    _reject_unknown(raw, _TOP_LEVEL, "config")
    schema_version = raw.get("schema_version")
    if schema_version != OPTIMIZATION_CONFIG_SCHEMA_VERSION:
        raise OptimizationConfigError(
            "schema_version must be " + OPTIMIZATION_CONFIG_SCHEMA_VERSION
        )
    if raw.get("kind") != "OptimizationConfig":
        raise OptimizationConfigError("kind must be OptimizationConfig")

    metadata_raw = _mapping(raw.get("metadata"), "metadata")
    _reject_unknown(metadata_raw, _METADATA, "metadata")
    config_id = metadata_raw.get("id")
    version = metadata_raw.get("version")
    if (
        not isinstance(config_id, str)
        or not config_id
        or not isinstance(version, str)
        or not version
    ):
        raise OptimizationConfigError(
            "metadata.id and metadata.version must be non-empty strings"
        )
    metadata = OptimizationConfigMetadata(config_id, version, str(metadata_raw.get("description", "")))

    model_raw = _mapping(raw.get("model", {}), "model")
    _reject_unknown(model_raw, _MODEL, "model")
    for name, value in model_raw.items():
        if not isinstance(value, str):
            raise OptimizationConfigError(f"model.{name} must be a string")
    compatibility_raw = _mapping(raw.get("compatibility", {}), "compatibility")
    _reject_unknown(compatibility_raw, _COMPATIBILITY, "compatibility")
    _validate_compatibility(compatibility_raw)

    extends_raw = raw.get("extends", [])
    if not isinstance(extends_raw, list) or not all(
        isinstance(item, str) and item for item in extends_raw
    ):
        raise OptimizationConfigError("extends must be a list of non-empty catalog IDs")

    modules_raw = raw.get("optimization_modules", [])
    if not isinstance(modules_raw, list) or not all(
        isinstance(item, str) and item for item in modules_raw
    ):
        raise OptimizationConfigError(
            "optimization_modules must be a list of non-empty Python module paths"
        )
    if len(set(modules_raw)) != len(modules_raw):
        raise OptimizationConfigError("optimization_modules must not contain duplicates")

    entries = []
    seen = set()
    groups_raw = raw.get("optimization_groups", [])
    if not isinstance(groups_raw, list):
        raise OptimizationConfigError("optimization_groups must be a list")
    for index, item in enumerate(groups_raw):
        path = f"optimization_groups[{index}]"
        item = _mapping(item, path)
        _reject_unknown(item, _GROUP, path)
        group_id = item.get("id")
        if not isinstance(group_id, str) or not group_id:
            raise OptimizationConfigError(f"{path}.id must be a non-empty string")
        if group_id in seen:
            raise OptimizationConfigError(f"duplicate OptimizationGroup entry: {group_id}")
        seen.add(group_id)
        trust = _mapping(item.get("trust", {}), f"{path}.trust")
        _reject_unknown(trust, _TRUST, f"{path}.trust")
        _validate_trust(trust, f"{path}.trust")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise OptimizationConfigError(f"{path}.enabled must be boolean")
        entries.append(
            OptimizationGroupConfig(
                id=group_id,
                enabled=enabled,
                options=FrozenDict(
                    _json_only(
                        _mapping(item.get("options", {}), f"{path}.options"), path
                    )
                ),
                trust=FrozenDict(_json_only(trust, f"{path}.trust")),
                _provided=tuple(sorted(item)),
            )
        )

    return OptimizationConfig(
        schema_version=schema_version,
        kind="OptimizationConfig",
        metadata=metadata,
        model=FrozenDict(_json_only(model_raw, "model")),
        extends=tuple(extends_raw),
        compatibility=FrozenDict(_json_only(compatibility_raw, "compatibility")),
        optimization_groups=tuple(entries),
        optimization_modules=tuple(modules_raw),
    )


def optimization_config_to_dict(config: OptimizationConfig) -> Dict[str, Any]:
    return to_primitive(config)
