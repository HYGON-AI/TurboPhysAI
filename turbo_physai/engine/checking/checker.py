# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from .evidence import ast_hash, source_hash
from ..execution.replacements import HandlerError, MechanismHandler
from ..execution.replacements.base import resolve_replacement
from .context import detect_target_context
from ..contracts import (
    CheckResult,
    CheckStatus,
    CompatibilityContext,
    CompatibilityResult,
    EnvironmentSnapshot,
    Mechanism,
    OptimizationGroup,
    ReplacementSpec,
    OptimizationGroupConfig,
)
from ..definitions.registry import Registry


def _result(
    code: str,
    status: CheckStatus,
    *,
    replacement_id: Optional[str] = None,
    expected: Any = None,
    actual: Any = None,
    overrideable: bool = False,
    detail: Optional[str] = None,
) -> CheckResult:
    return CheckResult(code, status, expected, actual, overrideable, detail, replacement_id)


def _target_type_check(value: Any, mechanism: Mechanism) -> Tuple[bool, str]:
    if mechanism == Mechanism.REPLACE:
        return (
            callable(value) or isinstance(value, property),
            "callable, class or property",
        )
    if mechanism == Mechanism.WRAPPER:
        return callable(value), "callable"
    return True, "module path"


def _signature_shape(value: Any) -> Any:
    if isinstance(value, property):
        return tuple(
            (name, _signature_shape(getattr(value, name)))
            for name in ("fget", "fset", "fdel")
            if getattr(value, name) is not None
        )
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return None
    return tuple(
        (
            parameter.name,
            parameter.kind.name,
            parameter.default is inspect.Parameter.empty,
        )
        for parameter in signature.parameters.values()
    )


def _representative_calls(signature: inspect.Signature):
    """Yield calls that cover the optional and argument-kind boundaries."""

    marker = object()
    parameters = tuple(signature.parameters.values())
    extra_keyword = "__turbo_physai_extra__"
    while extra_keyword in signature.parameters:
        extra_keyword += "_"
    for include_optional in (False, True):
        for keyword_style in (False, True):
            args = []
            kwargs = {}
            for parameter in parameters:
                required = parameter.default is inspect.Parameter.empty
                include = required or include_optional
                if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                    if include:
                        args.append(marker)
                elif parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if include:
                        if keyword_style:
                            kwargs[parameter.name] = marker
                        else:
                            args.append(marker)
                elif parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                    if include_optional and not keyword_style:
                        args.append(marker)
                elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                    if include:
                        kwargs[parameter.name] = marker
                elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    if include_optional:
                        kwargs[extra_keyword] = marker
            yield tuple(args), kwargs


def _signature_compatible(original: Any, replacement: Any) -> Optional[bool]:
    if isinstance(original, property) or isinstance(replacement, property):
        if not isinstance(original, property) or not isinstance(
            replacement, property
        ):
            return False
        for name in ("fget", "fset", "fdel"):
            original_accessor = getattr(original, name)
            replacement_accessor = getattr(replacement, name)
            if (original_accessor is None) != (replacement_accessor is None):
                return False
            if original_accessor is None:
                continue
            compatible = _signature_compatible(
                original_accessor,
                replacement_accessor,
            )
            if compatible is not True:
                return compatible
        return True
    try:
        original_signature = inspect.signature(original)
        replacement_signature = inspect.signature(replacement)
    except (TypeError, ValueError):
        return None
    for args, kwargs in _representative_calls(original_signature):
        try:
            replacement_signature.bind(*args, **kwargs)
        except TypeError:
            return False
    return True


def _trusted_values(
    trust: Mapping[str, Any], category: str, target: str
) -> Tuple[Any, ...]:
    values = trust.get(category, {})
    if isinstance(values, Mapping):
        target_values = values.get(target, ())
    else:
        target_values = values
    if isinstance(target_values, str):
        return (target_values,)
    if isinstance(target_values, Sequence):
        return tuple(target_values)
    return ()


def _values(value: Any) -> Tuple[Any, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _version_matches(actual: Optional[str], accepted: Sequence[str]) -> bool:
    if actual is None:
        return False
    for requirement in accepted:
        if not requirement.startswith(("<", ">", "=", "!", "~")):
            if actual == requirement:
                return True
            continue
        try:
            if Version(actual) in SpecifierSet(requirement):
                return True
        except (InvalidSpecifier, InvalidVersion):
            continue
    return False


class Checker:
    def __init__(
        self, registry: Registry, handlers: Mapping[Mechanism, MechanismHandler]
    ) -> None:
        self.registry = registry
        self.handlers = handlers
        self.prepared_groups: Dict[
            str, Tuple[Tuple[str, MechanismHandler, Any], ...]
        ] = {}

    def check_group(
        self,
        group: OptimizationGroup,
        entry: OptimizationGroupConfig,
        environment: EnvironmentSnapshot,
        *,
        import_missing: bool = True,
    ) -> Tuple[CheckResult, ...]:
        checks: List[CheckResult] = []
        prepared_units = []
        for replacement_id in group.members:
            spec = self.registry.get_spec(replacement_id)
            if spec is None:
                checks.append(
                    _result(
                        "registry.spec_missing",
                        CheckStatus.FAIL,
                        replacement_id=replacement_id,
                        expected=replacement_id,
                        detail="ReplacementSpec is not registered",
                    )
                )
                continue
            handler = self.handlers.get(spec.mechanism)
            if handler is None:
                checks.append(
                    _result(
                        "registry.handler_missing",
                        CheckStatus.FAIL,
                        replacement_id=replacement_id,
                        expected=spec.mechanism.value,
                    )
                )
                continue
            try:
                prepared = handler.prepare(
                    spec,
                    entry.options,
                    import_missing=import_missing,
                )
            except HandlerError as exc:
                detail = str(exc)
                if "runtime condition" in detail or "runtime_condition" in detail:
                    code = "runtime_condition.invalid"
                elif "aliases" in detail:
                    code = "alias.identity"
                elif "replacement" in detail:
                    code = "replacement.unresolved"
                else:
                    code = "target.unresolved"
                checks.append(
                    _result(code, CheckStatus.FAIL, replacement_id=replacement_id, detail=detail)
                )
                continue
            except Exception as exc:
                checks.append(
                    _result(
                        "replacement.load_error",
                        CheckStatus.FAIL,
                        replacement_id=replacement_id,
                        detail=str(exc),
                    )
                )
                continue

            prepared_units.append((replacement_id, handler, prepared))

            checked_spec = prepared.spec

            checks.append(
                _result("target.resolved", CheckStatus.PASS, replacement_id=replacement_id)
            )
            if checked_spec.mechanism == Mechanism.IMPORT_REPLACE:
                checks.extend(self._check_import(checked_spec))
                continue
            if checked_spec.mechanism in {
                Mechanism.IMPORT_ALIAS,
                Mechanism.OPTIONAL_IMPORT,
                Mechanism.REGISTRY_OVERRIDE,
            }:
                checks.append(
                    _result(
                        "import_compatibility.valid",
                        CheckStatus.PASS,
                        replacement_id=replacement_id,
                        expected=checked_spec.mechanism.value,
                        actual=checked_spec.target,
                    )
                )
                continue

            original = prepared.targets[0].original
            actual_type = f"{type(original).__module__}.{type(original).__qualname__}"
            type_ok, target_requirement = _target_type_check(
                original, checked_spec.mechanism
            )
            checks.append(
                _result(
                    "target.type",
                    CheckStatus.PASS if type_ok else CheckStatus.FAIL,
                    replacement_id=replacement_id,
                    expected=(
                        f"{checked_spec.mechanism.value} requires "
                        f"{target_requirement} target"
                    ),
                    actual=actual_type,
                )
            )
            evidence = self._source_evidence(original)
            signature_ok = _signature_compatible(original, prepared.implementation)
            native_artifact = str(evidence.get("source_hash") or "").startswith(
                "artifact-v1:"
            )
            checks.append(
                _result(
                    "target.signature",
                    (
                        (
                            CheckStatus.NOT_APPLICABLE
                            if native_artifact
                            else CheckStatus.UNKNOWN
                        )
                        if signature_ok is None
                        else (CheckStatus.PASS if signature_ok else CheckStatus.FAIL)
                    ),
                    replacement_id=replacement_id,
                    expected=_signature_shape(original),
                    actual=_signature_shape(prepared.implementation),
                )
            )
            if checked_spec.runtime_condition is not None:
                condition_signature_ok = _signature_compatible(
                    original,
                    prepared.runtime_condition,
                )
                checks.append(
                    _result(
                        "runtime_condition.signature",
                        (
                            CheckStatus.UNKNOWN
                            if condition_signature_ok is None
                            else (
                                CheckStatus.PASS
                                if condition_signature_ok
                                else CheckStatus.FAIL
                            )
                        ),
                        replacement_id=replacement_id,
                        expected=_signature_shape(original),
                        actual=_signature_shape(prepared.runtime_condition),
                        detail=checked_spec.runtime_condition,
                    )
                )
            # Every direct replacement and wrapper is bound to the target
            # source evidence captured by OptimizationConfig generation.
            checks.extend(
                self._source_identity_check(checked_spec, evidence, entry.trust)
            )
        if len(prepared_units) == len(group.members):
            checks.extend(
                self._compatibility_check(
                    group,
                    entry,
                    environment,
                    prepared_units,
                    import_missing=import_missing,
                )
            )
        self.prepared_groups[group.group_id] = tuple(prepared_units)
        return tuple(checks)

    def check_environment(
        self,
        environment: EnvironmentSnapshot,
        compatibility: Mapping[str, Any],
    ) -> Tuple[CheckResult, ...]:
        checks: List[CheckResult] = []
        expected_dependencies = compatibility.get("dependencies", {})
        if isinstance(expected_dependencies, Mapping):
            for name in sorted(expected_dependencies):
                expected = expected_dependencies[name]
                actual = environment.dependencies.get(name)
                accepted = (expected,) if isinstance(expected, str) else tuple(expected)
                checks.append(
                    _result(
                        "environment.dependency_version",
                        (
                            CheckStatus.PASS
                            if _version_matches(actual, accepted)
                            else CheckStatus.FAIL
                        ),
                        expected={name: accepted},
                        actual={name: actual},
                        overrideable=actual is not None,
                    )
                )
        commits = _values(compatibility.get("commits", ()))
        if commits:
            matches = environment.commit in commits
            checks.append(
                _result(
                    "project.commit",
                    CheckStatus.PASS if matches else CheckStatus.WARNING,
                    expected=commits,
                    actual=environment.commit,
                    detail=(
                        None
                        if matches
                        else "commit mismatch is informational; target evidence still decides applicability"
                    ),
                )
            )
        allow_dirty = compatibility.get("allow_dirty")
        if allow_dirty is not None:
            if bool(allow_dirty):
                dirty_status = CheckStatus.PASS
            elif environment.dirty is None:
                dirty_status = CheckStatus.UNKNOWN
            else:
                dirty_status = (
                    CheckStatus.FAIL if environment.dirty else CheckStatus.PASS
                )
            checks.append(
                _result(
                    "project.dirty",
                    dirty_status,
                    expected=bool(allow_dirty),
                    actual=environment.dirty,
                    overrideable=True,
                )
            )
        backend = compatibility.get("backend")
        if backend:
            accepted = (backend,) if isinstance(backend, str) else tuple(backend)
            checks.append(
                _result(
                    "environment.backend",
                    (
                        CheckStatus.PASS
                        if environment.backend in accepted
                        else CheckStatus.FAIL
                    ),
                    expected=accepted,
                    actual=environment.backend,
                )
            )
        repository = compatibility.get("repository")
        if repository:
            accepted = (
                (repository,) if isinstance(repository, str) else tuple(repository)
            )
            checks.append(
                _result(
                    "project.repository",
                    (
                        CheckStatus.PASS
                        if environment.repository in accepted
                        else CheckStatus.FAIL
                    ),
                    expected=accepted,
                    actual=environment.repository,
                )
            )
        return tuple(checks)

    @staticmethod
    def _source_evidence(original: Any) -> Dict[str, Optional[str]]:
        return {
            "source_hash": source_hash(original),
            "ast_hash": ast_hash(original),
        }

    def _source_identity_check(
        self,
        spec: ReplacementSpec,
        actual: Mapping[str, Optional[str]],
        trust: Mapping[str, Any],
    ) -> Iterable[CheckResult]:
        trusted_source = _trusted_values(trust, "source_hashes", spec.target)
        trusted_ast = _trusted_values(trust, "ast_hashes", spec.target)
        expected = {
            "source_hashes": trusted_source,
            "ast_hashes": trusted_ast,
        }
        if not any(actual.values()):
            status = CheckStatus.UNKNOWN
        elif not trusted_source and not trusted_ast:
            status = CheckStatus.NOT_APPLICABLE
        else:
            accepted = (
                actual.get("source_hash") in trusted_source
                or actual.get("ast_hash") in trusted_ast
            )
            status = CheckStatus.PASS if accepted else CheckStatus.FAIL
        return (
            _result(
                "source.identity",
                status,
                replacement_id=spec.replacement_id,
                expected=expected,
                actual=actual,
                overrideable=any(actual.values()),
            ),
        )

    def _check_import(self, spec: ReplacementSpec) -> Iterable[CheckResult]:
        import sys

        loaded = spec.target in sys.modules
        return (
            _result(
                "import.timing",
                CheckStatus.FAIL if loaded else CheckStatus.PASS,
                replacement_id=spec.replacement_id,
                expected="module not loaded",
                actual="loaded" if loaded else "not_loaded",
            ),
        )

    def _compatibility_check(
        self,
        group: OptimizationGroup,
        entry: OptimizationGroupConfig,
        environment: EnvironmentSnapshot,
        prepared_units: Sequence[Tuple[str, MechanismHandler, Any]],
        *,
        import_missing: bool,
    ) -> Iterable[CheckResult]:
        path = group.compatibility_check
        if path is None:
            return ()
        try:
            check = resolve_replacement(path, import_missing=import_missing)
        except HandlerError as exc:
            return (
                _result(
                    "compatibility.unresolved",
                    CheckStatus.FAIL,
                    expected=path,
                    detail=str(exc),
                ),
            )
        if not callable(check) or isinstance(check, type):
            return (
                _result(
                    "compatibility.invalid_check",
                    CheckStatus.FAIL,
                    expected="non-class callable",
                    actual=f"{type(check).__module__}.{type(check).__qualname__}",
                    detail=path,
                ),
            )
        targets = tuple(
            detect_target_context(prepared.spec.target, prepared.targets[0].original)
            for _, _, prepared in prepared_units
            if prepared.targets
        )
        context = CompatibilityContext(
            group.group_id,
            environment,
            targets,
            entry.options,
        )
        try:
            result = check(context)
        except Exception as exc:
            return (
                _result(
                    "compatibility.error",
                    CheckStatus.FAIL,
                    expected="CompatibilityResult",
                    actual=type(exc).__name__,
                    detail=f"{path}: {exc}",
                ),
            )
        if not isinstance(result, CompatibilityResult):
            return (
                _result(
                    "compatibility.invalid_result",
                    CheckStatus.FAIL,
                    expected="CompatibilityResult",
                    actual=f"{type(result).__module__}.{type(result).__qualname__}",
                    detail=path,
                ),
            )
        return (
            _result(
                "compatibility.custom",
                CheckStatus.PASS if result.compatible else CheckStatus.FAIL,
                expected=result.expected,
                actual=result.actual,
                detail=result.reason or path,
            ),
        )
