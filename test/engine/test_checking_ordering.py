# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import _bz2
import sys
import types
import unittest

from turbo_physai.engine.checking.evidence import ast_hash, source_hash
from turbo_physai.engine.contracts import (
    CheckResult,
    CheckStatus,
    CompatibilityResult,
    Decision,
    EnvironmentSnapshot,
    FrozenDict,
    Mechanism,
    OptimizationConfig,
    OptimizationGroup,
    ReplacementSpec,
    OptimizationGroupConfig,
    OptimizationConfigMetadata,
)
from turbo_physai.engine.checking.ordering import Preparation, _decide
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.execution.replacements import default_handlers


def original(value):
    return value + 1


def replacement(value):
    return value + 2


def wrapper(original, options):
    del options
    return original


def positive(value):
    return value > 0


def condition_without_arguments():
    return True


def compatible_group(context):
    return CompatibilityResult(
        compatible=True,
        expected="supported",
        actual=context.group_id,
    )


def incompatible_group(context):
    return CompatibilityResult(
        compatible=False,
        expected="supported baseline",
        actual=context.targets[0].commit,
        reason="target repository baseline is not supported",
    )


class NativeLikeCallable:
    """Callable with a native module artifact and no inspectable signature."""

    __module__ = _bz2.__name__
    __signature__ = object()

    def __call__(self, *args, **kwargs):
        return args, kwargs


native_callable = NativeLikeCallable()


def context():
    return EnvironmentSnapshot("3.10", "test", "python", "/tmp")


def config(*entries):
    return OptimizationConfig(
        "turbophysai/optimization-config/v1",
        "OptimizationConfig",
        OptimizationConfigMetadata("demo", "1"),
        optimization_groups=tuple(entries),
    )


class CheckingOrderingTest(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("optimization_engine_fake")
        self.module.original = original
        self.module.other = original
        self.module.replacement = replacement
        self.module.wrapper = wrapper
        self.module.positive = positive
        self.module.condition_without_arguments = condition_without_arguments
        self.module.compatible_group = compatible_group
        self.module.incompatible_group = incompatible_group
        self.module.original_replacement = original
        self.module.import_module = types.ModuleType("optimization_engine_import")
        self.module.import_module.VALUE = 1
        self.module.import_module.__file__ = __file__
        self.module.native = native_callable
        sys.modules[self.module.__name__] = self.module

    def tearDown(self):
        sys.modules.pop(self.module.__name__, None)

    def registry(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "demo.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.original",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("demo.group", ("demo.replacement",)))
        return registry

    def prepare(
        self,
        registry,
        model_optimization_config,
        force_groups=(),
    ):
        return Preparation(registry, default_handlers()).prepare(
            run_id="run",
            config=model_optimization_config,
            environment=context(),
            force_groups=force_groups,
        )

    def test_valid_group_applies(self):
        registry = self.registry()
        prepared_execution = self.prepare(
            registry,
            config(OptimizationGroupConfig("demo.group")),
        )
        self.assertEqual(prepared_execution.groups[0].decision, Decision.APPLY)
        self.assertEqual(prepared_execution.execution_order, ("demo.group",))

    def test_warning_check_does_not_block(self):
        decision, reason, force_used = _decide(
            checks=(CheckResult("project.commit", CheckStatus.WARNING),)
        )
        self.assertEqual(decision, Decision.APPLY)
        self.assertEqual(reason, "checks_passed")
        self.assertFalse(force_used)

    def test_commit_mismatch_is_reported_without_blocking_group(self):
        registry = self.registry()
        model_optimization_config = OptimizationConfig(
            "turbophysai/optimization-config/v1",
            "OptimizationConfig",
            OptimizationConfigMetadata("demo", "1"),
            compatibility=FrozenDict({"commits": ["expected"]}),
            optimization_groups=(OptimizationGroupConfig("demo.group"),),
        )
        prepared_execution = Preparation(registry, default_handlers()).prepare(
            run_id="run",
            config=model_optimization_config,
            environment=EnvironmentSnapshot(
                "3.10",
                "test",
                "python",
                "/tmp",
                commit="actual",
            ),
        )
        group = prepared_execution.groups[0]
        commit_check = next(
            check for check in prepared_execution.checks if check.code == "project.commit"
        )
        self.assertEqual(commit_check.status, CheckStatus.WARNING)
        self.assertFalse(
            any(check.code == "project.commit" for check in group.checks)
        )
        self.assertEqual(group.decision, Decision.APPLY)

    def test_runtime_condition_signature_is_checked(self):
        for condition, expected_status, expected_decision in (
            (
                "optimization_engine_fake.positive",
                CheckStatus.PASS,
                Decision.APPLY,
            ),
            (
                "optimization_engine_fake.condition_without_arguments",
                CheckStatus.FAIL,
                Decision.BLOCK,
            ),
        ):
            with self.subTest(condition=condition):
                registry = Registry()
                registry.register_spec(
                    ReplacementSpec(
                        "conditional.replacement",
                        Mechanism.REPLACE,
                        "optimization_engine_fake.original",
                        "optimization_engine_fake.replacement",
                        runtime_condition=condition,
                    )
                )
                registry.register_group(
                    OptimizationGroup("conditional.group", ("conditional.replacement",))
                )
                prepared_execution = self.prepare(
                    registry,
                    config(OptimizationGroupConfig("conditional.group")),
                )
                check = next(
                    item
                    for item in prepared_execution.groups[0].checks
                    if item.code == "runtime_condition.signature"
                )
                self.assertEqual(check.status, expected_status)
                self.assertEqual(
                    prepared_execution.groups[0].decision,
                    expected_decision,
                )

    def test_ast_identity_can_accept_format_only_source_change(self):
        registry = self.registry()
        entry = OptimizationGroupConfig(
            "demo.group",
            trust=FrozenDict(
                {
                    "source_hashes": {"optimization_engine_fake.original": "source-v1:stale"},
                    "ast_hashes": {"optimization_engine_fake.original": ast_hash(original)},
                }
            ),
        )
        prepared_execution = self.prepare(registry, config(entry))
        identity = next(
            check
            for check in prepared_execution.groups[0].checks
            if check.code == "source.identity"
        )
        self.assertEqual(identity.status, CheckStatus.PASS)
        self.assertEqual(prepared_execution.groups[0].decision, Decision.APPLY)

    def test_native_callable_uses_artifact_identity_without_signature(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "native.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.native",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("native.group", ("native.replacement",)))
        entry = OptimizationGroupConfig(
            "native.group",
            trust=FrozenDict(
                {
                    "source_hashes": {
                        "optimization_engine_fake.native": source_hash(native_callable)
                    }
                }
            ),
        )

        prepared_execution = self.prepare(registry, config(entry))
        signature = next(
            check
            for check in prepared_execution.groups[0].checks
            if check.code == "target.signature"
        )
        identity = next(
            check
            for check in prepared_execution.groups[0].checks
            if check.code == "source.identity"
        )
        self.assertEqual(signature.status, CheckStatus.NOT_APPLICABLE)
        self.assertEqual(identity.status, CheckStatus.PASS)
        self.assertEqual(prepared_execution.groups[0].decision, Decision.APPLY)

    def test_standard_model_wrapper_still_checks_target_identity(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "wrapper.replacement",
                Mechanism.WRAPPER,
                "optimization_engine_fake.original",
                "optimization_engine_fake.wrapper",
            )
        )
        registry.register_group(OptimizationGroup("wrapper.group", ("wrapper.replacement",)))
        entry = OptimizationGroupConfig(
            "wrapper.group",
            trust=FrozenDict(
                {
                    "source_hashes": {
                        "optimization_engine_fake.original": "source-v1:stale"
                    },
                    "ast_hashes": {
                        "optimization_engine_fake.original": "ast-v1:stale"
                    },
                }
            ),
        )

        prepared_execution = self.prepare(registry, config(entry))
        identity = next(
            check
            for check in prepared_execution.groups[0].checks
            if check.code == "source.identity"
        )

        self.assertEqual(identity.status, CheckStatus.FAIL)
        self.assertEqual(prepared_execution.groups[0].decision, Decision.BLOCK)

        forced = self.prepare(
            registry,
            config(entry),
            force_groups=("wrapper.group",),
        )
        self.assertEqual(forced.groups[0].decision, Decision.APPLY)
        self.assertTrue(forced.groups[0].forced)

    def test_disabled_group_skips(self):
        registry = self.registry()
        prepared_execution = self.prepare(
            registry, config(OptimizationGroupConfig("demo.group", enabled=False))
        )
        self.assertEqual(prepared_execution.groups[0].decision, Decision.SKIP)
        self.assertEqual(prepared_execution.execution_order, ())

    def test_missing_target_blocks(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "missing.replacement",
                Mechanism.WRAPPER,
                "optimization_engine_fake.missing",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("missing.group", ("missing.replacement",)))
        prepared_execution = self.prepare(registry, config(OptimizationGroupConfig("missing.group")))
        self.assertEqual(prepared_execution.groups[0].decision, Decision.BLOCK)

        forced = self.prepare(
            registry,
            config(OptimizationGroupConfig("missing.group")),
            force_groups=("missing.group",),
        )
        self.assertEqual(forced.groups[0].decision, Decision.BLOCK)

    def test_replace_blocks_mismatched_replacement_type(self):
        registry = self.registry()
        self.module.not_a_function = type
        registry.register_spec(
            ReplacementSpec(
                "wrong.type.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.not_a_function",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("wrong.type.group", ("wrong.type.replacement",)))
        prepared_execution = self.prepare(registry, config(OptimizationGroupConfig("wrong.type.group")))
        check = next(
            item
            for item in prepared_execution.groups[0].checks
            if item.replacement_id == "wrong.type.replacement"
        )
        self.assertEqual(check.status, CheckStatus.FAIL)
        self.assertIn("class replacement must be a class", check.detail)
        self.assertEqual(prepared_execution.groups[0].decision, Decision.BLOCK)

    def test_different_replacements_on_same_target_are_blocked(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "other.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.original",
                "optimization_engine_fake.original_replacement",
            )
        )
        registry.register_group(OptimizationGroup("other.group", ("other.replacement",)))
        prepared_execution = self.prepare(
            registry, config(OptimizationGroupConfig("demo.group"), OptimizationGroupConfig("other.group"))
        )
        self.assertTrue(
            all(item.decision == Decision.BLOCK for item in prepared_execution.groups)
        )

    def test_incompatible_effects_within_group_are_blocked_with_evidence(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "same_group.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.original",
                "optimization_engine_fake.original_replacement",
            )
        )
        registry.register_group(
            OptimizationGroup(
                "conflicted.group", ("demo.replacement", "same_group.replacement")
            )
        )
        prepared_execution = self.prepare(
            registry, config(OptimizationGroupConfig("conflicted.group"))
        )
        self.assertEqual(prepared_execution.groups[0].decision, Decision.BLOCK)
        check = next(
            item
            for item in prepared_execution.groups[0].checks
            if item.code == "target.intra_group_conflict"
        )
        self.assertEqual(
            check.actual["replacement_ids"], ("demo.replacement", "same_group.replacement")
        )
        self.assertEqual(
            check.actual["replacements"],
            (
                "optimization_engine_fake.replacement",
                "optimization_engine_fake.original_replacement",
            ),
        )

    def test_unconstrained_groups_follow_config_declaration_order(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "other.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.other",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("z.group", ("demo.replacement",)))
        registry.register_group(OptimizationGroup("a.group", ("other.replacement",)))
        prepared_execution = self.prepare(
            registry, config(OptimizationGroupConfig("z.group"), OptimizationGroupConfig("a.group"))
        )
        self.assertEqual(prepared_execution.execution_order, ("z.group", "a.group"))

    def test_dependency_order_is_stable(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "other.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.other",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(
            OptimizationGroup("other.group", ("other.replacement",), depends_on=("demo.group",))
        )
        prepared_execution = self.prepare(
            registry, config(OptimizationGroupConfig("other.group"), OptimizationGroupConfig("demo.group"))
        )
        self.assertEqual(prepared_execution.execution_order, ("demo.group", "other.group"))

    def test_blocked_dependency_propagates_to_dependent_group(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "blocked.replacement",
                Mechanism.WRAPPER,
                "optimization_engine_fake.missing",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "dependent.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.other",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("blocked.group", ("blocked.replacement",)))
        registry.register_group(
            OptimizationGroup(
                "dependent.group",
                ("dependent.replacement",),
                depends_on=("blocked.group",),
            )
        )
        prepared_execution = self.prepare(
            registry,
            config(OptimizationGroupConfig("blocked.group"), OptimizationGroupConfig("dependent.group")),
        )
        self.assertEqual(prepared_execution.groups[0].decision, Decision.BLOCK)
        self.assertEqual(prepared_execution.groups[1].decision, Decision.BLOCK)
        self.assertIn("dependency.blocked", {c.code for c in prepared_execution.groups[1].checks})

    def test_source_mismatch_requires_explicit_force(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "critical.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.original",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(OptimizationGroup("critical.group", ("critical.replacement",)))
        model_optimization_config = config(
            OptimizationGroupConfig(
                "critical.group",
                trust=FrozenDict(
                    {
                        "source_hashes": {
                            "optimization_engine_fake.original": "source-v1:stale"
                        }
                    }
                ),
            )
        )
        blocked = self.prepare(registry, model_optimization_config)
        identity_check = next(
            check
            for check in blocked.groups[0].checks
            if check.code == "source.identity"
        )
        self.assertEqual(identity_check.status, CheckStatus.FAIL)
        self.assertEqual(blocked.groups[0].decision, Decision.BLOCK)
        allowed = self.prepare(
            registry,
            model_optimization_config,
            force_groups=("critical.group",),
        )
        self.assertEqual(allowed.groups[0].decision, Decision.APPLY)
        self.assertTrue(allowed.groups[0].forced)

    def test_dependency_cycle_blocks_groups(self):
        registry = self.registry()
        registry.register_spec(
            ReplacementSpec(
                "other.replacement",
                Mechanism.REPLACE,
                "optimization_engine_fake.other",
                "optimization_engine_fake.replacement",
            )
        )
        registry.register_group(
            OptimizationGroup("other.group", ("other.replacement",), depends_on=("demo.group",))
        )
        # Replace the original group with a fresh registry to create the opposite edge.
        cycled = Registry()
        for spec in registry.specs.values():
            cycled.register_spec(spec)
        cycled.register_group(
            OptimizationGroup("demo.group", ("demo.replacement",), depends_on=("other.group",))
        )
        cycled.register_group(
            OptimizationGroup("other.group", ("other.replacement",), depends_on=("demo.group",))
        )
        prepared_execution = self.prepare(
            cycled, config(OptimizationGroupConfig("demo.group"), OptimizationGroupConfig("other.group"))
        )
        self.assertTrue(
            all(item.decision == Decision.BLOCK for item in prepared_execution.groups)
        )

    def test_import_replace_does_not_use_replacement_hash_evidence(self):
        sys.modules.pop("planned_import.op", None)
        sys.modules.pop("planned_import", None)
        registry = Registry()

        registry.register_spec(
            ReplacementSpec(
                "import.replacement",
                Mechanism.IMPORT_REPLACE,
                "planned_import.op",
                "optimization_engine_fake.import_module",
            )
        )
        registry.register_group(OptimizationGroup("import.group", ("import.replacement",)))
        model_optimization_config = config(
            OptimizationGroupConfig(
                "import.group",
                trust=FrozenDict(
                    {
                        "source_hashes": {
                            "optimization_engine_fake.import_module": "source-v1:stale"
                        }
                    }
                ),
            )
        )
        prepared_execution = self.prepare(registry, model_optimization_config)
        self.assertNotIn(
            "replacement.identity",
            {check.code for check in prepared_execution.groups[0].checks},
        )
        self.assertEqual(prepared_execution.groups[0].decision, Decision.APPLY)
        self.assertFalse(prepared_execution.groups[0].forced)

    def test_group_compatibility_check_runs_once_before_execution(self):
        for check_path, expected_status, expected_decision in (
            (
                "optimization_engine_fake.compatible_group",
                CheckStatus.PASS,
                Decision.APPLY,
            ),
            (
                "optimization_engine_fake.incompatible_group",
                CheckStatus.FAIL,
                Decision.BLOCK,
            ),
        ):
            with self.subTest(check_path=check_path):
                registry = Registry()
                registry.register_spec(
                    ReplacementSpec(
                        "compatibility.replacement",
                        Mechanism.REPLACE,
                        "optimization_engine_fake.original",
                        "optimization_engine_fake.replacement",
                    )
                )
                registry.register_group(
                    OptimizationGroup(
                        "compatibility.group",
                        ("compatibility.replacement",),
                        compatibility_check=check_path,
                    )
                )
                prepared_execution = self.prepare(
                    registry,
                    config(OptimizationGroupConfig("compatibility.group")),
                )
                compatibility = [
                    item
                    for item in prepared_execution.groups[0].checks
                    if item.code == "compatibility.custom"
                ]
                self.assertEqual(len(compatibility), 1)
                self.assertEqual(compatibility[0].status, expected_status)
                self.assertEqual(prepared_execution.groups[0].decision, expected_decision)
                if expected_status == CheckStatus.FAIL:
                    forced = self.prepare(
                        registry,
                        config(OptimizationGroupConfig("compatibility.group")),
                        force_groups=("compatibility.group",),
                    )
                    self.assertEqual(
                        forced.groups[0].decision,
                        Decision.BLOCK,
                    )


if __name__ == "__main__":
    unittest.main()
