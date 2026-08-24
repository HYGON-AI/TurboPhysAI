# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import json
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import turbo_physai
import turbo_physai.engine as engine_module
from turbo_physai.engine.errors import OptimizationConfigError
from turbo_physai.engine.contracts import (
    Decision,
    ExecutionStatus,
    Mechanism,
    OptimizationGroup,
    ReplacementSpec,
)
from turbo_physai.engine.definitions.registry import Registry


def original(value):
    return value + 1


def replacement(value):
    return value + 10


def wrapper(function, options):
    def wrapped(value):
        return function(value) * options.get("factor", 2)

    return wrapped


def positive(value):
    return value > 0


def write_optimization_config(path: Path, *, group="demo.group"):
    path.write_text(
        textwrap.dedent(
            f"""
            schema_version: turbophysai/optimization-config/v1
            kind: OptimizationConfig
            metadata: {{id: demo, version: "1"}}
            optimization_groups:
              - id: {group}
                options: {{factor: 3}}
            """
        ),
        encoding="utf-8",
    )


class EngineExecutorTest(unittest.TestCase):
    def setUp(self):
        engine_module._apply_called = False
        self.module = types.ModuleType("engine_fake")
        self.module.original = original
        self.module.alias = original
        self.module.replacement = replacement
        self.module.wrapper = wrapper
        self.module.positive = positive
        sys.modules[self.module.__name__] = self.module

    def tearDown(self):
        sys.modules.pop(self.module.__name__, None)

    def registry(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "demo.replacement",
                Mechanism.WRAPPER,
                "engine_fake.original",
                "engine_fake.wrapper",
                aliases=("engine_fake.alias",),
            )
        )
        registry.register_group(OptimizationGroup("demo.group", ("demo.replacement",)))
        return registry

    def test_check_does_not_mutate_target_or_modules(self):
        registry = self.registry()
        with tempfile.TemporaryDirectory() as directory:
            optimization_config_path = Path(directory) / "config.yaml"
            write_optimization_config(optimization_config_path)
            before_modules = dict(sys.modules)
            before = self.module.original
            resolved = turbo_physai.check(optimization_config_path=optimization_config_path, registry=registry)
        self.assertIs(self.module.original, before)
        self.assertIs(self.module.alias, before)
        self.assertEqual(set(sys.modules), set(before_modules))
        self.assertEqual(resolved.execution_order, ("demo.group",))

    def test_public_api_exposes_check_instead_of_inspect(self):
        self.assertTrue(callable(turbo_physai.check))
        self.assertFalse(hasattr(turbo_physai, "inspect"))

    def test_public_api_no_longer_accepts_execution_mode(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'mode'"):
            turbo_physai.check(mode="safe")

    def test_check_force_groups_directly_allows_overrideable_check(self):
        registry = self.registry()
        with tempfile.TemporaryDirectory() as directory:
            optimization_config_path = Path(directory) / "config.yaml"
            optimization_config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: demo, version: "1"}
                    optimization_groups:
                      - id: demo.group
                        options: {factor: 3}
                        trust:
                          source_hashes:
                            engine_fake.original: ["source-v1:stale"]
                          ast_hashes:
                            engine_fake.original: ["ast-v1:stale"]
                    """
                ),
                encoding="utf-8",
            )
            blocked = turbo_physai.check(optimization_config_path=optimization_config_path, registry=registry)
            forced = turbo_physai.check(
                optimization_config_path=optimization_config_path,
                registry=registry,
                force_groups=("demo.group",),
            )

        self.assertEqual(blocked.groups[0].decision, Decision.BLOCK)
        self.assertEqual(forced.groups[0].decision, Decision.APPLY)
        self.assertTrue(forced.groups[0].forced)

    def test_force_groups_rejects_group_not_enabled_by_config(self):
        registry = self.registry()
        with tempfile.TemporaryDirectory() as directory:
            optimization_config_path = Path(directory) / "config.yaml"
            write_optimization_config(optimization_config_path)
            with self.assertRaisesRegex(
                OptimizationConfigError,
                "must reference enabled OptimizationGroups",
            ):
                turbo_physai.check(
                    optimization_config_path=optimization_config_path,
                    registry=registry,
                    force_groups=("other.group",),
                )

    def test_apply_changes_target_and_writes_report(self):
        registry = self.registry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            write_optimization_config(optimization_config_path)
            report = turbo_physai.apply(
                optimization_config_path=optimization_config_path,
                registry=registry,
                report_dir=root / "reports",
            )
            self.assertEqual(self.module.original(2), 9)
            self.assertEqual(self.module.alias(2), 9)
            self.assertEqual(report.execution[0].status, ExecutionStatus.APPLIED)
            self.assertTrue(Path(report.artifacts.json_path).is_file())
            payload = json.loads(
                Path(report.artifacts.json_path).read_text(encoding="utf-8")
            )
            self.assertEqual(payload["summary"]["applied"], 1)
            self.assertEqual(
                payload["optimization_config"]["metadata"]["id"], "demo"
            )
            self.assertNotIn("model_optimization_config", payload)
            self.assertEqual(
                payload["optimization_config_path"], str(optimization_config_path.resolve())
            )
            self.assertNotIn("model_optimization_config_path", payload)
            self.assertIsNone(payload["runtime_config_path"])
            self.assertNotIn("mode", payload["prepared_execution"])
            markdown = Path(report.artifacts.markdown_path).read_text(encoding="utf-8")
            self.assertNotIn("- Mode:", markdown)
            self.assertIn("- OptimizationConfig: `demo` `1`", markdown)
            self.assertIn(
                f"- OptimizationConfig path: `{optimization_config_path.resolve()}`", markdown
            )
            self.assertIn("## Configuration Checks", markdown)
            self.assertIn("OptimizationConfig path", markdown)
            self.assertIn("- RuntimeConfig path: not used", markdown)

    def test_apply_installs_runtime_condition_dispatcher(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "conditional.replacement",
                Mechanism.REPLACE,
                "engine_fake.original",
                "engine_fake.replacement",
                aliases=("engine_fake.alias",),
                runtime_condition="engine_fake.positive",
            )
        )
        registry.register_group(
            OptimizationGroup("demo.group", ("conditional.replacement",))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            write_optimization_config(optimization_config_path)
            report = turbo_physai.apply(
                optimization_config_path=optimization_config_path,
                registry=registry,
                report_dir=root / "reports",
            )
        self.assertEqual(report.execution[0].status, ExecutionStatus.APPLIED)
        self.assertIs(self.module.original, self.module.alias)
        self.assertEqual(self.module.original(-2), -1)
        self.assertEqual(self.module.original(2), 12)

    def test_apply_evaluates_fixed_wrapper_once(self):
        calls = []

        def counting_wrapper(function, options):
            calls.append("demo.replacement")
            return wrapper(function, options)

        registry = Registry()
        self.module.counting_wrapper = counting_wrapper
        registry.register_spec(
            ReplacementSpec(
                "demo.replacement",
                Mechanism.WRAPPER,
                "engine_fake.original",
                "engine_fake.counting_wrapper",
                aliases=("engine_fake.alias",),
            )
        )
        registry.register_group(OptimizationGroup("demo.group", ("demo.replacement",)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            write_optimization_config(optimization_config_path)
            turbo_physai.apply(
                optimization_config_path=optimization_config_path,
                registry=registry,
                report_dir=root / "reports",
            )
        self.assertEqual(calls, ["demo.replacement"])

    def test_apply_rejects_second_call_in_same_process(self):
        registry = self.registry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            write_optimization_config(optimization_config_path)
            turbo_physai.apply(
                optimization_config_path=optimization_config_path,
                registry=registry,
                report_dir=root / "first",
            )
            with self.assertRaisesRegex(
                OptimizationConfigError, "only be called once per process"
            ):
                turbo_physai.apply(
                    optimization_config_path=optimization_config_path,
                    registry=registry,
                    report_dir=root / "second",
                )
        self.assertEqual(self.module.original(2), 9)

    def test_block_writes_report_without_mutation(self):
        registry = Registry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            write_optimization_config(optimization_config_path, group="missing.group")
            report = turbo_physai.apply(
                optimization_config_path=optimization_config_path,
                registry=registry,
                report_dir=root / "reports",
            )
            self.assertEqual(report.summary["blocked"], 1)
            self.assertTrue(Path(report.artifacts.json_path).is_file())
            with self.assertRaisesRegex(
                OptimizationConfigError, "only be called once per process"
            ):
                turbo_physai.apply(
                    optimization_config_path=optimization_config_path,
                    registry=registry,
                    report_dir=root / "retry",
                )

    def test_block_does_not_stop_independent_group(self):
        registry = self.registry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            optimization_config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: demo, version: "1"}
                    optimization_groups:
                      - {id: missing.group}
                      - {id: demo.group, options: {factor: 3}}
                    """
                ),
                encoding="utf-8",
            )
            report = turbo_physai.apply(
                optimization_config_path=optimization_config_path, registry=registry, report_dir=root
            )
        self.assertEqual(report.summary["blocked"], 1)
        self.assertEqual(report.summary["applied"], 1)
        self.assertEqual(self.module.original(2), 9)

if __name__ == "__main__":
    unittest.main()
