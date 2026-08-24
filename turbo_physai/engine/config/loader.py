# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import yaml

from ..errors import OptimizationConfigError, OptimizationConfigNotFoundError
from ..contracts import OptimizationConfig, OptimizationGroupConfig
from .schema import optimization_config_from_dict


PACKAGED_OPTIMIZATION_ROOT = Path(__file__).resolve().parents[2] / "optimizations"
PACKAGED_MODEL_OPTIMIZATION_ROOT = PACKAGED_OPTIMIZATION_ROOT / "models"
PACKAGED_DEFAULT_OPTIMIZATION_CONFIG = (
    PACKAGED_OPTIMIZATION_ROOT / "common" / "configs" / "optimization.yaml"
)


class OptimizationConfigCatalog:
    def __init__(self, configs: Optional[Mapping[str, OptimizationConfig]] = None) -> None:
        self._configs: Dict[str, OptimizationConfig] = dict(configs or {})

    def register(self, config: OptimizationConfig) -> OptimizationConfig:
        config_id = config.metadata.id
        if config_id in self._configs:
            raise OptimizationConfigError(
                f"duplicate OptimizationConfig ID: {config_id}"
            )
        self._configs[config_id] = config
        return config

    def get(self, config_id: str) -> Optional[OptimizationConfig]:
        return self._configs.get(config_id)

    @classmethod
    def from_builtin_files(cls) -> "OptimizationConfigCatalog":
        catalog = cls()
        for path in sorted(PACKAGED_OPTIMIZATION_ROOT.glob("**/configs/optimization.yaml")):
            # macOS archive extraction can leave AppleDouble metadata beside
            # real files (for example ``._optimization.yaml``).
            # They are binary metadata, never OptimizationConfig files.
            if path.name.startswith("."):
                continue
            config = _load_yaml(path)
            if config.metadata.id != "default":
                catalog.register(config)
        return catalog


def _packaged_model_optimization_config(model: str) -> Path:
    model_name = model.strip().lower().replace("-", "_")
    if not model_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in model_name
    ):
        raise OptimizationConfigError(f"invalid model name: {model!r}")
    candidate = (
        PACKAGED_MODEL_OPTIMIZATION_ROOT
        / model_name
        / "configs"
        / "optimization.yaml"
    )
    if not candidate.is_file():
        available = (
            sorted(
                path.name
                for path in PACKAGED_MODEL_OPTIMIZATION_ROOT.iterdir()
                if (path / "configs" / "optimization.yaml").is_file()
            )
            if PACKAGED_MODEL_OPTIMIZATION_ROOT.is_dir()
            else []
        )
        choices = ", ".join(available) if available else "none"
        raise OptimizationConfigNotFoundError(
            f"unknown built-in model {model!r}; available models: {choices}"
        )
    return candidate


def resolve_optimization_config_path(
    optimization_config_path: Optional[os.PathLike] = None,
    *,
    model: Optional[str] = None,
) -> Path:
    if optimization_config_path is not None:
        candidate = Path(optimization_config_path)
    elif model is not None:
        candidate = _packaged_model_optimization_config(model)
    elif os.environ.get("TURBO_PHYSAI_OPTIMIZATION_CONFIG"):
        candidate = Path(os.environ["TURBO_PHYSAI_OPTIMIZATION_CONFIG"])
    else:
        conventional = (
            Path.cwd() / "turbophysai_configs" / "default" / "optimization.yaml"
        )
        candidate = (
            conventional
            if conventional.is_file()
            else PACKAGED_DEFAULT_OPTIMIZATION_CONFIG
        )
    candidate = candidate.expanduser().resolve()
    if not candidate.is_file():
        raise OptimizationConfigNotFoundError(f"OptimizationConfig not found: {candidate}")
    return candidate


def _load_yaml(path: Path) -> OptimizationConfig:
    try:
        with path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise OptimizationConfigError(
            f"failed to read OptimizationConfig {path}: {exc}"
        ) from exc
    if raw is None:
        raise OptimizationConfigError(f"OptimizationConfig is empty: {path}")
    return optimization_config_from_dict(raw)


def _merge(base: OptimizationConfig, overlay: OptimizationConfig) -> OptimizationConfig:
    entries: Dict[str, OptimizationGroupConfig] = {
        entry.id: entry for entry in base.optimization_groups
    }
    order = [entry.id for entry in base.optimization_groups]
    for entry in overlay.optimization_groups:
        if entry.id not in entries:
            order.append(entry.id)
            entries[entry.id] = entry
            continue
        previous = entries[entry.id]
        provided = set(entry._provided)
        options = dict(previous.options.items())
        if "options" in provided:
            options.update(entry.options.items())
        trust = dict(previous.trust.items())
        if "trust" in provided:
            for category, value in entry.trust.items():
                if isinstance(value, Mapping) and isinstance(
                    trust.get(category), Mapping
                ):
                    merged_category = dict(trust[category].items())
                    merged_category.update(value.items())
                    trust[category] = merged_category
                else:
                    trust[category] = value
        entries[entry.id] = OptimizationGroupConfig(
            id=entry.id,
            enabled=entry.enabled if "enabled" in provided else previous.enabled,
            options=options,
            trust=trust,
            _provided=tuple(sorted(set(previous._provided) | provided)),
        )
    model = dict(base.model.items())
    model.update(overlay.model.items())
    compatibility = dict(base.compatibility.items())
    compatibility.update(overlay.compatibility.items())
    optimization_modules = tuple(
        dict.fromkeys(base.optimization_modules + overlay.optimization_modules)
    )
    return OptimizationConfig(
        schema_version=overlay.schema_version,
        kind=overlay.kind,
        metadata=overlay.metadata,
        model=model,
        extends=(),
        compatibility=compatibility,
        optimization_groups=tuple(entries[group_id] for group_id in order),
        optimization_modules=optimization_modules,
    )


def _resolve_extends(
    config: OptimizationConfig, catalog: OptimizationConfigCatalog, stack: Tuple[str, ...] = ()
) -> OptimizationConfig:
    merged = OptimizationConfig(
        schema_version=config.schema_version,
        kind=config.kind,
        metadata=config.metadata,
    )
    for config_id in config.extends:
        if config_id in stack:
            raise OptimizationConfigError(
                f"OptimizationConfig inheritance cycle: "
                f"{' -> '.join(stack + (config_id,))}"
            )
        parent = catalog.get(config_id)
        if parent is None:
            raise OptimizationConfigError(
                f"unknown OptimizationConfig ID: {config_id}"
            )
        merged = _merge(
            merged, _resolve_extends(parent, catalog, stack + (config_id,))
        )
    return _merge(merged, config)


def load_optimization_config(
    optimization_config_path: Optional[os.PathLike] = None,
    *,
    catalog: Optional[OptimizationConfigCatalog] = None,
) -> OptimizationConfig:
    config = _load_yaml(resolve_optimization_config_path(optimization_config_path))
    return resolve_optimization_config(config, catalog=catalog)


def resolve_optimization_config(
    config: OptimizationConfig, *, catalog: Optional[OptimizationConfigCatalog] = None
) -> OptimizationConfig:
    resolved = _resolve_extends(
        config, catalog or OptimizationConfigCatalog.from_builtin_files(), (config.metadata.id,)
    )
    for module_name in resolved.optimization_modules:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise OptimizationConfigError(
                f"failed to import optimization module {module_name}: {exc}"
            ) from exc
    return resolved
