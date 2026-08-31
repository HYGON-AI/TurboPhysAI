# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import io
import sys
import tempfile
import textwrap
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from turbo_physai.cli import main as cli_main
import turbo_physai.engine as engine_module
from turbo_physai.engine.checking.context import detect_context
from turbo_physai.engine.execution.executor import Executor
from turbo_physai.engine.contracts import (
    ExecutionStatus,
    Mechanism,
    OptimizationConfig,
    OptimizationGroup,
    ReplacementSpec,
    OptimizationGroupConfig,
    OptimizationConfigMetadata,
    RestoreResult,
    RestoreStatus,
)
from turbo_physai.engine.checking.ordering import Preparation
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.execution.replacements import (
    ReplaceHandler,
    WrapperHandler,
    default_handlers,
)


def first(value):
    return value + 1


def second(value):
    return value + 2


def third(value):
    return value + 3


def wrapper(function, options):
    del options

    def wrapped(value):
        return function(value) * 10

    return wrapped


class FailingWrapperHandler(WrapperHandler):
    def apply(self, prepared):
        if prepared.spec.replacement_id == "second.replacement":
            raise RuntimeError("injected apply failure")
        return super().apply(prepared)


class FailedRestoreWrapperHandler(FailingWrapperHandler):
    def restore(self, snapshot):
        return tuple(
            RestoreResult(item.path, RestoreStatus.FAILED, "injected restore failure")
            for item in snapshot
        )


class FailingPropertyReplaceHandler(ReplaceHandler):
    def apply(self, prepared):
        if prepared.spec.replacement_id == "property.second":
            raise RuntimeError("injected apply failure")
        return super().apply(prepared)


class TransactionsAndCliTest(unittest.TestCase):
    def setUp(self):
        engine_module._apply_called = False
        self.module = types.ModuleType("transaction_fake")
        self.module.first = first
        self.module.first_alias = first
        self.module.second = second
        self.module.third = third
        self.module.wrapper = wrapper
        self.module.second_replacement = second
        sys.modules[self.module.__name__] = self.module

    def tearDown(self):
        sys.modules.pop(self.module.__name__, None)

    def registry(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "first.replacement",
                Mechanism.WRAPPER,
                "transaction_fake.first",
                "transaction_fake.wrapper",
                aliases=("transaction_fake.first_alias",),
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "second.replacement",
                Mechanism.WRAPPER,
                "transaction_fake.second",
                "transaction_fake.wrapper",
            )
        )
        registry.register_group(
            OptimizationGroup("transaction.group", ("first.replacement", "second.replacement"))
        )
        return registry

    def prepared_execution(self, registry, handlers):
        config = OptimizationConfig(
            "turbophysai/optimization-config/v1",
            "OptimizationConfig",
            OptimizationConfigMetadata("transaction", "1"),
            optimization_groups=(OptimizationGroupConfig("transaction.group"),),
        )
        preparation = Preparation(registry, handlers)
        prepared_execution = preparation.prepare(
            run_id="run",
            config=config,
            environment=detect_context(),
        )
        return config, prepared_execution, preparation.prepared_groups

    def test_group_failure_restores_target_and_alias(self):
        registry = self.registry()
        handlers = default_handlers()
        handlers[Mechanism.WRAPPER] = FailingWrapperHandler()
        config, prepared_execution, prepared = self.prepared_execution(registry, handlers)
        outcome = Executor().execute(prepared_execution, prepared_groups=prepared)
        self.assertIs(self.module.first, first)
        self.assertIs(self.module.first_alias, first)
        self.assertIs(self.module.second, second)
        self.assertEqual(outcome.groups[0].status, ExecutionStatus.ROLLED_BACK)
        self.assertIsNone(outcome.terminal_error)

    def test_group_failure_restores_property(self):
        class PropertyTarget:
            @property
            def value(self):
                return 1

        def optimized_value(self):
            return 2

        original_property = PropertyTarget.value
        self.module.PropertyTarget = PropertyTarget
        self.module.optimized_property = property(optimized_value)
        self.module.third_replacement = third

        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "property.first",
                Mechanism.REPLACE,
                "transaction_fake.PropertyTarget.value",
                "transaction_fake.optimized_property",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "property.second",
                Mechanism.REPLACE,
                "transaction_fake.third",
                "transaction_fake.third_replacement",
            )
        )
        registry.register_group(
            OptimizationGroup(
                "property.group",
                ("property.first", "property.second"),
            )
        )
        config = OptimizationConfig(
            "turbophysai/optimization-config/v1",
            "OptimizationConfig",
            OptimizationConfigMetadata("property", "1"),
            optimization_groups=(OptimizationGroupConfig("property.group"),),
        )
        handlers = default_handlers()
        handlers[Mechanism.REPLACE] = FailingPropertyReplaceHandler()
        preparation = Preparation(registry, handlers)
        prepared_execution = preparation.prepare(
            run_id="run",
            config=config,
            environment=detect_context(),
        )

        outcome = Executor().execute(
            prepared_execution,
            prepared_groups=preparation.prepared_groups,
        )

        self.assertIs(PropertyTarget.value, original_property)
        self.assertEqual(PropertyTarget().value, 1)
        self.assertEqual(outcome.groups[0].status, ExecutionStatus.ROLLED_BACK)

    def test_runtime_dependency_failure_skips_only_downstream_group(self):
        class PrerequisiteFailingHandler(WrapperHandler):
            def apply(self, prepared):
                if prepared.spec.replacement_id == "prerequisite.replacement":
                    raise RuntimeError("injected prerequisite failure")
                return super().apply(prepared)

        registry = Registry()
        for replacement_id, target in (
            ("prerequisite.replacement", "transaction_fake.first"),
            ("dependent.replacement", "transaction_fake.second"),
            ("independent.replacement", "transaction_fake.third"),
        ):
            registry.register_spec(
                ReplacementSpec(
                    replacement_id,
                    Mechanism.WRAPPER,
                    target,
                    "transaction_fake.wrapper",
                )
            )
        registry.register_group(
            OptimizationGroup("prerequisite.group", ("prerequisite.replacement",))
        )
        registry.register_group(
            OptimizationGroup(
                "dependent.group",
                ("dependent.replacement",),
                depends_on=("prerequisite.group",),
            )
        )
        registry.register_group(
            OptimizationGroup("independent.group", ("independent.replacement",))
        )
        config = OptimizationConfig(
            "turbophysai/optimization-config/v1",
            "OptimizationConfig",
            OptimizationConfigMetadata("dependencies", "1"),
            optimization_groups=(
                OptimizationGroupConfig("prerequisite.group"),
                OptimizationGroupConfig("dependent.group"),
                OptimizationGroupConfig("independent.group"),
            ),
        )
        handlers = default_handlers()
        handlers[Mechanism.WRAPPER] = PrerequisiteFailingHandler()
        preparation = Preparation(registry, handlers)
        prepared_execution = preparation.prepare(
            run_id="run",
            config=config,
            environment=detect_context(),
        )

        outcome = Executor().execute(
            prepared_execution,
            prepared_groups=preparation.prepared_groups,
        )

        self.assertEqual(
            tuple(result.status for result in outcome.groups),
            (
                ExecutionStatus.ROLLED_BACK,
                ExecutionStatus.NOT_STARTED,
                ExecutionStatus.APPLIED,
            ),
        )
        self.assertIn("prerequisite.group", outcome.groups[1].error)
        self.assertIsNone(outcome.terminal_error)

    def test_rollback_failure_is_terminal(self):
        registry = self.registry()
        handlers = default_handlers()
        handlers[Mechanism.WRAPPER] = FailedRestoreWrapperHandler()
        config, prepared_execution, prepared = self.prepared_execution(registry, handlers)
        outcome = Executor().execute(prepared_execution, prepared_groups=prepared)
        self.assertTrue(outcome.rollback_failed)
        self.assertIsNotNone(outcome.terminal_error)
        self.assertEqual(outcome.groups[0].status, ExecutionStatus.FAILED)

    def test_silent_restore_mismatch_is_terminal(self):
        class SilentRestoreMapping(dict):
            def __setitem__(self, key, value):
                if key == "first" and value is first:
                    return
                super().__setitem__(key, value)

        values = SilentRestoreMapping()
        dict.__setitem__(values, "first", first)
        dict.__setitem__(values, "second", second)
        module = types.ModuleType("silent_restore_fake")
        module.values = values
        module.wrapper = wrapper
        sys.modules[module.__name__] = module

        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "first.replacement",
                Mechanism.WRAPPER,
                "silent_restore_fake.values.first",
                "silent_restore_fake.wrapper",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "second.replacement",
                Mechanism.WRAPPER,
                "silent_restore_fake.values.second",
                "silent_restore_fake.wrapper",
            )
        )
        registry.register_group(
            OptimizationGroup("transaction.group", ("first.replacement", "second.replacement"))
        )
        handlers = default_handlers()
        handlers[Mechanism.WRAPPER] = FailingWrapperHandler()
        try:
            config, prepared_execution, prepared = self.prepared_execution(registry, handlers)
            outcome = Executor().execute(
                prepared_execution, prepared_groups=prepared
            )
        finally:
            sys.modules.pop(module.__name__, None)

        self.assertTrue(outcome.rollback_failed)
        self.assertIsNotNone(outcome.terminal_error)
        self.assertEqual(outcome.groups[0].status, ExecutionStatus.FAILED)
        self.assertTrue(
            any(
                result.status == RestoreStatus.FAILED
                and "identity does not match" in (result.error or "")
                for result in outcome.groups[0].rollback_results
            )
        )

    def test_executor_rejects_missing_prepared_group(self):
        registry = self.registry()
        handlers = default_handlers()
        config, prepared_execution, _ = self.prepared_execution(registry, handlers)
        outcome = Executor().execute(prepared_execution, prepared_groups={})
        self.assertEqual(outcome.groups[0].status, ExecutionStatus.FAILED)
        self.assertIn("does not match", outcome.groups[0].error)

    def test_cli_validate_and_diff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_config = root / "first.yaml"
            second_config = root / "second.yaml"
            content = textwrap.dedent(
                """
                schema_version: turbophysai/optimization-config/v1
                kind: OptimizationConfig
                metadata: {id: cli, version: "1"}
                optimization_groups: []
                """
            )
            first_config.write_text(content, encoding="utf-8")
            second_config.write_text(
                content.replace('version: "1"', 'version: "2"'), encoding="utf-8"
            )
            validation_output = io.StringIO()
            with redirect_stdout(validation_output):
                self.assertEqual(
                    cli_main(["optimization", "validate", str(first_config)]), 0
                )
            self.assertEqual(
            validation_output.getvalue(), "valid OptimizationConfig: cli 1\n"
            )
            self.assertEqual(
                cli_main(["optimization", "diff", str(first_config), str(second_config)]), 0
            )

    def test_cli_does_not_expose_obsolete_inspect(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli_main(["optimization", "inspect", "config.yaml"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())

    def test_cli_optimization_check_validates_generated_config(self):
        checked = OptimizationConfig(
            "turbophysai/optimization-config/v1",
            "OptimizationConfig",
            OptimizationConfigMetadata("checked", "1"),
        )
        stdout = io.StringIO()
        with patch(
            "turbo_physai.engine.config.generator.check_optimization_config",
            return_value=checked,
        ) as check_optimization_config, redirect_stdout(stdout):
            result = cli_main(
                [
                    "optimization",
                    "check",
                    "generated.yaml",
                    "--repo",
                    "model-repo",
                ]
            )
        self.assertEqual(result, 0)
        check_optimization_config.assert_called_once_with(
            Path("generated.yaml"), Path("model-repo")
        )
        self.assertEqual(
            stdout.getvalue(), "checked OptimizationConfig: checked 1\n"
        )

    def test_cli_generate_no_longer_accepts_check_flag(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            cli_main(
                [
                    "optimization",
                    "generate",
                    "--recipe",
                    "template.yaml",
                    "--repo",
                    "model-repo",
                    "--commit",
                    "abc123",
                    "--output",
                    "generated.yaml",
                    "--check",
                ]
            )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --check", stderr.getvalue())

if __name__ == "__main__":
    unittest.main()
