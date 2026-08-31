# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ...contracts import Mechanism, ReplacementSpec, RestoreResult, RestoreStatus
from .base import HandlerError, MechanismHandler, PreparedReplacement, resolve_replacement


def _registry_mapping(registry: Any) -> Any:
    mapping = getattr(registry, "module_dict", None)
    if mapping is None:
        mapping = getattr(registry, "_module_dict", None)
    if mapping is None or not hasattr(mapping, "clear") or not hasattr(mapping, "update"):
        raise HandlerError("registry must expose a mutable module_dict mapping")
    return mapping


@dataclass(frozen=True)
class RegistryOverrideSnapshot:
    module_name: str
    registry_path: str
    registry: Any
    entries: Tuple[Tuple[str, Any], ...]
    modules: Tuple[Tuple[str, Any], ...]
    register_was_local: bool
    local_register: Any


class RegistryOverrideHandler(MechanismHandler):
    mechanism = Mechanism.REGISTRY_OVERRIDE

    def prepare(
        self,
        spec: ReplacementSpec,
        options: Mapping[str, Any],
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        del options
        registry = resolve_replacement(
            spec.replacement, import_missing=import_missing
        )
        _registry_mapping(registry)
        if not callable(getattr(registry, "_register_module", None)):
            raise HandlerError(
                f"registry does not support _register_module: {spec.replacement}"
            )
        names = tuple(spec.mechanism_options.get("names", ()))
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise HandlerError(
                f"registry override names are required: {spec.replacement_id}"
            )
        return PreparedReplacement(spec, (), (registry, names))

    def snapshot(self, prepared: PreparedReplacement) -> RegistryOverrideSnapshot:
        registry, _ = prepared.replacement
        mapping = _registry_mapping(registry)
        return RegistryOverrideSnapshot(
            prepared.spec.target,
            prepared.spec.replacement,
            registry,
            tuple(mapping.items()),
            tuple(sys.modules.items()),
            "_register_module" in getattr(registry, "__dict__", {}),
            getattr(registry, "__dict__", {}).get("_register_module"),
        )

    @staticmethod
    def _requested_names(module: Any, module_name: Any) -> Tuple[str, ...]:
        if module_name is None:
            return (module.__name__,)
        if isinstance(module_name, str):
            return (module_name,)
        return tuple(module_name)

    def apply(self, prepared: PreparedReplacement) -> Tuple[str, ...]:
        registry, allowed_names = prepared.replacement
        allowed = frozenset(allowed_names)
        mapping = _registry_mapping(registry)

        if prepared.spec.target in sys.modules:
            invalid = [
                name
                for name in allowed_names
                if name not in mapping
                or getattr(mapping[name], "__module__", None) != prepared.spec.target
            ]
            if invalid:
                raise HandlerError(
                    "registry module was imported without expected overrides: "
                    + ", ".join(invalid)
                )
            return tuple(f"{prepared.spec.replacement}.{name}" for name in allowed_names)

        original_register = registry._register_module
        register_was_local = "_register_module" in getattr(registry, "__dict__", {})
        local_register = getattr(registry, "__dict__", {}).get("_register_module")

        def restricted_register(*args: Any, **kwargs: Any) -> Any:
            module = kwargs.get("module_class", kwargs.get("module"))
            if module is None and args:
                module = args[0]
            module_name = kwargs.get("module_name")
            if module_name is None and len(args) > 1:
                module_name = args[1]
            names = self._requested_names(module, module_name)
            eligible = (
                getattr(module, "__module__", None) == prepared.spec.target
                and set(names).issubset(allowed)
            )
            if eligible:
                if len(args) > 2:
                    args = tuple(args[:2]) + (True,) + tuple(args[3:])
                else:
                    kwargs["force"] = True
            return original_register(*args, **kwargs)

        registry._register_module = restricted_register
        try:
            importlib.import_module(prepared.spec.target)
        finally:
            if register_was_local:
                registry.__dict__["_register_module"] = local_register
            else:
                registry.__dict__.pop("_register_module", None)

        invalid = [
            name
            for name in allowed_names
            if name not in mapping
            or getattr(mapping[name], "__module__", None) != prepared.spec.target
        ]
        if invalid:
            raise HandlerError(
                "registry import did not override all declared names: "
                + ", ".join(invalid)
            )
        return tuple(f"{prepared.spec.replacement}.{name}" for name in allowed_names)

    def restore(
        self, snapshot: RegistryOverrideSnapshot
    ) -> Tuple[RestoreResult, ...]:
        results = []
        try:
            mapping = _registry_mapping(snapshot.registry)
            mapping.clear()
            mapping.update(snapshot.entries)
            if snapshot.register_was_local:
                snapshot.registry.__dict__["_register_module"] = snapshot.local_register
            else:
                snapshot.registry.__dict__.pop("_register_module", None)
            results.append(
                RestoreResult(snapshot.registry_path, RestoreStatus.RESTORED)
            )
        except Exception as exc:
            results.append(
                RestoreResult(snapshot.registry_path, RestoreStatus.FAILED, str(exc))
            )
        try:
            original_modules = dict(snapshot.modules)
            for name in tuple(sys.modules):
                if name not in original_modules:
                    sys.modules.pop(name, None)
            for name, module in original_modules.items():
                sys.modules[name] = module
            results.append(
                RestoreResult(snapshot.module_name, RestoreStatus.RESTORED)
            )
        except Exception as exc:
            results.append(
                RestoreResult(snapshot.module_name, RestoreStatus.FAILED, str(exc))
            )
        return tuple(results)
