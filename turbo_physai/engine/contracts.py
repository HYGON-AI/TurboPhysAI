# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Iterator, Optional, Tuple


REPORT_SCHEMA_VERSION = "turbophysai.optimization-report/v1alpha1"
PREPARED_EXECUTION_SCHEMA_VERSION = "turbophysai/prepared-execution/v1"
OPTIMIZATION_CONFIG_SCHEMA_VERSION = "turbophysai/optimization-config/v1"


class FrozenDict(Mapping):
    """Small immutable mapping used by frozen public data contracts."""

    __slots__ = ("_data",)

    def __init__(self, value: Optional[Mapping[str, Any]] = None) -> None:
        self._data = {
            str(key): freeze_json(item) for key, item in (value or {}).items()
        }

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False


def freeze_json(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


def to_primitive(value: Any) -> Any:
    """Convert optimization engine objects into deterministic JSON-compatible values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: to_primitive(getattr(value, item.name))
            for item in fields(value)
            if not item.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {str(key): to_primitive(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    return value


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Mechanism(StrEnum):
    REPLACE = "replace"
    WRAPPER = "wrapper"
    IMPORT_REPLACE = "import_replace"
    IMPORT_ALIAS = "import_alias"
    OPTIONAL_IMPORT = "optional_import"
    REGISTRY_OVERRIDE = "registry_override"


class CheckStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class Decision(StrEnum):
    APPLY = "apply"
    SKIP = "skip"
    BLOCK = "block"


class ExecutionStatus(StrEnum):
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    NOT_STARTED = "not_started"


class RestoreStatus(StrEnum):
    RESTORED = "restored"
    FAILED = "failed"


@dataclass(frozen=True)
class ReplacementSpec:
    replacement_id: str
    mechanism: Mechanism
    target: str
    replacement: str
    aliases: Tuple[str, ...] = ()
    runtime_condition: Optional[str] = None
    mechanism_options: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        if (
            not self.replacement_id
            or not self.target
            or not self.replacement
        ):
            raise ValueError("replacement_id, target and replacement are required")
        if self.runtime_condition is not None and not self.runtime_condition:
            raise ValueError("runtime_condition must be a non-empty object path")
        if len(set((self.target,) + tuple(self.aliases))) != 1 + len(
            tuple(self.aliases)
        ):
            raise ValueError("target and aliases must be unique")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(
            self, "mechanism_options", freeze_json(self.mechanism_options)
        )


@dataclass(frozen=True)
class OptimizationGroup:
    group_id: str
    members: Tuple[str, ...]
    depends_on: Tuple[str, ...] = ()
    compatibility_check: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.group_id or not self.members:
            raise ValueError("group_id and members are required")
        if len(set(self.members)) != len(self.members):
            raise ValueError("OptimizationGroup members must be unique")
        if isinstance(self.depends_on, str) or any(
            not isinstance(item, str) or not item for item in self.depends_on
        ):
            raise ValueError("OptimizationGroup dependencies must be non-empty strings")
        if self.group_id in self.depends_on:
            raise ValueError("OptimizationGroup cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("OptimizationGroup dependencies must be unique")
        if self.compatibility_check is not None and not self.compatibility_check:
            raise ValueError("compatibility_check must be a non-empty object path")
        for name in ("members", "depends_on"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class OptimizationConfigMetadata:
    id: str
    version: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.version:
            raise ValueError(
                "OptimizationConfig metadata id and version are required"
            )


@dataclass(frozen=True)
class OptimizationGroupConfig:
    id: str
    enabled: bool = True
    options: FrozenDict = field(default_factory=FrozenDict)
    trust: FrozenDict = field(default_factory=FrozenDict)
    _provided: Tuple[str, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("OptimizationGroupConfig id is required")
        object.__setattr__(self, "options", freeze_json(self.options))
        object.__setattr__(self, "trust", freeze_json(self.trust))
        object.__setattr__(self, "_provided", tuple(self._provided))


@dataclass(frozen=True)
class OptimizationConfig:
    schema_version: str
    kind: str
    metadata: OptimizationConfigMetadata
    model: FrozenDict = field(default_factory=FrozenDict)
    extends: Tuple[str, ...] = ()
    compatibility: FrozenDict = field(default_factory=FrozenDict)
    optimization_groups: Tuple[OptimizationGroupConfig, ...] = ()
    optimization_modules: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.schema_version != OPTIMIZATION_CONFIG_SCHEMA_VERSION
            or self.kind != "OptimizationConfig"
        ):
            raise ValueError(
                "unsupported OptimizationConfig schema_version or kind"
            )
        object.__setattr__(self, "model", freeze_json(self.model))
        object.__setattr__(self, "extends", tuple(self.extends))
        object.__setattr__(self, "compatibility", freeze_json(self.compatibility))
        object.__setattr__(self, "optimization_groups", tuple(self.optimization_groups))
        object.__setattr__(
            self, "optimization_modules", tuple(self.optimization_modules)
        )


@dataclass(frozen=True)
class EnvironmentSnapshot:
    python_version: str
    platform: str
    executable: str
    cwd: str
    dependencies: FrozenDict = field(default_factory=FrozenDict)
    repository: Optional[str] = None
    commit: Optional[str] = None
    dirty: Optional[bool] = None
    backend: Optional[str] = None
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependencies", freeze_json(self.dependencies))


@dataclass(frozen=True)
class CompatibilityTarget:
    path: str
    source_file: Optional[str] = None
    repository_root: Optional[str] = None
    repository: Optional[str] = None
    commit: Optional[str] = None
    dirty: Optional[bool] = None


@dataclass(frozen=True)
class CompatibilityContext:
    group_id: str
    environment: EnvironmentSnapshot
    targets: Tuple[CompatibilityTarget, ...]
    options: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "options", freeze_json(self.options))

    @staticmethod
    def package_version(distribution: str) -> Optional[str]:
        """Return an installed distribution version for an explicit check."""

        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(distribution)
        except PackageNotFoundError:
            return None


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    expected: Any = None
    actual: Any = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.compatible, bool):
            raise ValueError("CompatibilityResult.compatible must be boolean")
        object.__setattr__(self, "expected", freeze_json(self.expected))
        object.__setattr__(self, "actual", freeze_json(self.actual))


@dataclass(frozen=True)
class CheckResult:
    code: str
    status: CheckStatus
    expected: Any = None
    actual: Any = None
    overrideable: bool = False
    detail: Optional[str] = None
    replacement_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("CheckResult code is required")
        object.__setattr__(self, "expected", freeze_json(self.expected))
        object.__setattr__(self, "actual", freeze_json(self.actual))


@dataclass(frozen=True)
class Conflict:
    code: str
    groups: Tuple[str, ...]
    target: Optional[str] = None
    detail: Optional[str] = None
    replacement_ids: Tuple[str, ...] = ()
    replacements: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "replacement_ids", tuple(self.replacement_ids))
        object.__setattr__(self, "replacements", tuple(self.replacements))


@dataclass(frozen=True)
class PreparedGroup:
    group_id: str
    depends_on: Tuple[str, ...]
    members: Tuple[str, ...]
    checks: Tuple[CheckResult, ...]
    decision: Decision
    reason: str
    forced: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "checks", tuple(self.checks))


@dataclass(frozen=True)
class PreparedExecution:
    run_id: str
    environment: EnvironmentSnapshot
    groups: Tuple[PreparedGroup, ...]
    conflicts: Tuple[Conflict, ...]
    execution_order: Tuple[str, ...]
    checks: Tuple[CheckResult, ...] = ()
    schema_version: str = PREPARED_EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(self, "execution_order", tuple(self.execution_order))
        object.__setattr__(self, "checks", tuple(self.checks))


@dataclass(frozen=True)
class RestoreResult:
    path: str
    status: RestoreStatus
    error: Optional[str] = None


@dataclass(frozen=True)
class ReplacementResult:
    replacement_id: str
    status: ExecutionStatus
    changed_targets: Tuple[str, ...] = ()
    error: Optional[str] = None
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed_targets", tuple(self.changed_targets))


@dataclass(frozen=True)
class GroupExecutionResult:
    group_id: str
    status: ExecutionStatus
    members: Tuple[ReplacementResult, ...] = ()
    rollback_results: Tuple[RestoreResult, ...] = ()
    forced: bool = False
    error: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "members", tuple(self.members))
        object.__setattr__(self, "rollback_results", tuple(self.rollback_results))


@dataclass(frozen=True)
class ReportArtifacts:
    json_path: Optional[str] = None
    markdown_path: Optional[str] = None


@dataclass(frozen=True)
class OptimizationReport:
    run_id: str
    optimization_config: FrozenDict
    prepared_execution: PreparedExecution
    execution: Tuple[GroupExecutionResult, ...]
    summary: FrozenDict
    optimization_config_path: Optional[str] = None
    runtime_config_path: Optional[str] = None
    artifacts: ReportArtifacts = field(default_factory=ReportArtifacts)
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "optimization_config", freeze_json(self.optimization_config)
        )
        object.__setattr__(self, "execution", tuple(self.execution))
        object.__setattr__(self, "summary", freeze_json(self.summary))
