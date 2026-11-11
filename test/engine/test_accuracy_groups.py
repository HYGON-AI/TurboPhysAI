"""accuracy_only Group 开关：精度验证专用 Group 只在该模式下应用。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from turbo_physai.bootstrap import ACCURACY_MODE, bootstrap_environment
from turbo_physai.engine import _apply_group_overrides
from turbo_physai.engine.config.loader import _merge
from turbo_physai.engine.config.schema import optimization_config_from_dict
from turbo_physai.engine.contracts import (
    OptimizationConfig,
    OptimizationConfigMetadata,
    OptimizationGroupConfig,
)
from turbo_physai.engine.errors import OptimizationConfigError, TurboPhysAIError
from turbo_physai.runtime import RuntimeConfig, load_runtime_config, prepare_environment


class _EmptyRegistry:
    def get_group(self, group_id):
        return None


def _config(*groups):
    return OptimizationConfig(
        "turbophysai/optimization-config/v1",
        "OptimizationConfig",
        OptimizationConfigMetadata("demo", "1"),
        optimization_groups=tuple(groups),
    )


def _enabled(config):
    return {entry.id for entry in config.optimization_groups if entry.enabled}


class AccuracyOnlyGroupTest(unittest.TestCase):
    def setUp(self):
        self.config = _config(
            OptimizationGroupConfig("demo.speed"),
            OptimizationGroupConfig("demo.order", accuracy_only=True),
        )
        self.registry = _EmptyRegistry()

    def test_accuracy_mode_applies_accuracy_only_group(self):
        effective, _, _ = _apply_group_overrides(
            self.config, self.registry, (), (), accuracy_mode=True
        )
        self.assertEqual(_enabled(effective), {"demo.speed", "demo.order"})

    def test_disabled_accuracy_mode_skips_accuracy_only_group(self):
        effective, _, reasons = _apply_group_overrides(
            self.config, self.registry, (), (), accuracy_mode=False
        )
        self.assertEqual(_enabled(effective), {"demo.speed"})
        self.assertEqual(reasons["demo.order"], "accuracy_only_group")

    def test_forcing_accuracy_only_group_without_accuracy_mode_raises(self):
        with self.assertRaises(OptimizationConfigError) as raised:
            _apply_group_overrides(
                self.config, self.registry, ("demo.order",), (), accuracy_mode=False
            )
        self.assertIn("accuracy-only", str(raised.exception))

    def test_forcing_accuracy_only_group_with_accuracy_mode_is_allowed(self):
        effective, forced, _ = _apply_group_overrides(
            self.config, self.registry, ("demo.order",), (), accuracy_mode=True
        )
        self.assertEqual(forced, ("demo.order",))
        self.assertIn("demo.order", _enabled(effective))

    def test_accuracy_only_defaults_to_false(self):
        self.assertFalse(OptimizationGroupConfig("demo.speed").accuracy_only)

    def test_default_mode_skips_accuracy_only_group(self):
        effective, _, reasons = _apply_group_overrides(
            self.config, self.registry, (), ()
        )
        self.assertEqual(_enabled(effective), {"demo.speed"})
        self.assertEqual(reasons["demo.order"], "accuracy_only_group")


class AccuracyOnlySchemaTest(unittest.TestCase):
    def test_schema_accepts_accuracy_only(self):
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "demo", "version": "1"},
                "optimization_groups": [
                    {"id": "demo.order", "accuracy_only": True},
                ],
            }
        )
        self.assertTrue(config.optimization_groups[0].accuracy_only)

    def test_schema_rejects_non_boolean_accuracy_only(self):
        with self.assertRaises(Exception):
            optimization_config_from_dict(
                {
                    "schema_version": "turbophysai/optimization-config/v1",
                    "kind": "OptimizationConfig",
                    "metadata": {"id": "demo", "version": "1"},
                    "optimization_groups": [
                        {"id": "demo.order", "accuracy_only": "yes"},
                    ],
                }
            )


class AccuracyOnlyMergeTest(unittest.TestCase):
    def _config(self, groups):
        return optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "demo", "version": "1"},
                "optimization_groups": groups,
            }
        )

    def test_overlay_keeps_base_accuracy_only_when_not_provided(self):
        base = self._config([{"id": "demo.order", "accuracy_only": True}])
        overlay = self._config([{"id": "demo.order", "options": {"window": 20}}])
        merged = _merge(base, overlay).optimization_groups[0]
        self.assertTrue(merged.accuracy_only)
        self.assertEqual(dict(merged.options), {"window": 20})

    def test_overlay_can_clear_accuracy_only(self):
        base = self._config([{"id": "demo.order", "accuracy_only": True}])
        overlay = self._config([{"id": "demo.order", "accuracy_only": False}])
        self.assertFalse(_merge(base, overlay).optimization_groups[0].accuracy_only)


class AccuracyModeEnvironmentTest(unittest.TestCase):
    def test_environment_marks_accuracy_mode(self):
        self.assertEqual(
            bootstrap_environment({}, optimization_config="c", accuracy_mode=True)[
                ACCURACY_MODE
            ],
            "1",
        )
        self.assertNotIn(
            ACCURACY_MODE,
            bootstrap_environment({}, optimization_config="c", accuracy_mode=False),
        )
        self.assertNotIn(ACCURACY_MODE, bootstrap_environment({}, optimization_config="c"))


class AccuracyRuntimeEnvironmentTest(unittest.TestCase):
    def _runtime(self):
        return RuntimeConfig(
            {"BASE": "1"},
            (),
            {},
            {},
            True,
            {"ACCURACY_ONLY": "yes"},
            ("BASE",),
        )

    def test_load_runtime_config_reads_accuracy_section(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "runtime.yaml"
            path.write_text(
                "schema_version: turbophysai/runtime-config/v1\n"
                "kind: RuntimeConfig\n"
                "environment:\n"
                "  set:\n"
                "    BASE: '1'\n"
                "  accuracy:\n"
                "    set:\n"
                "      ACCURACY_ONLY: 'yes'\n"
                "    unset:\n"
                "      - BASE\n",
                encoding="utf-8",
            )
            runtime = load_runtime_config(path)
        self.assertEqual(dict(runtime.accuracy_set), {"ACCURACY_ONLY": "yes"})
        self.assertEqual(runtime.accuracy_unset, ("BASE",))

    def test_accuracy_environment_applies_in_accuracy_mode(self):
        with patch.dict(os.environ, {"BASE": "shell"}, clear=True):
            env = prepare_environment(self._runtime(), accuracy_mode=True)
        self.assertEqual(env["ACCURACY_ONLY"], "yes")
        self.assertNotIn("BASE", env)

    def test_accuracy_environment_is_ignored_by_default(self):
        with patch.dict(os.environ, {"BASE": "shell"}, clear=True):
            env = prepare_environment(self._runtime())
        self.assertEqual(env["BASE"], "1")
        self.assertNotIn("ACCURACY_ONLY", env)

    def test_command_line_set_wins_over_accuracy_section(self):
        with patch.dict(os.environ, {"BASE": "shell"}, clear=True):
            env = prepare_environment(
                self._runtime(),
                accuracy_mode=True,
                overrides={"ACCURACY_ONLY": "cli"},
            )
        self.assertEqual(env["ACCURACY_ONLY"], "cli")


class AccuracyOnlyCliValidationTest(unittest.TestCase):
    def _write_config(self, folder):
        path = Path(folder) / "optimization.yaml"
        path.write_text(
            "schema_version: turbophysai/optimization-config/v1\n"
            "kind: OptimizationConfig\n"
            "metadata:\n"
            "  id: demo\n"
            "  version: '1'\n"
            "optimization_groups:\n"
            "  - id: demo.order\n"
            "    accuracy_only: true\n",
            encoding="utf-8",
        )
        return path

    def test_cli_rejects_forcing_accuracy_only_group_without_accuracy(self):
        from turbo_physai.cli import _validate_run_groups

        with tempfile.TemporaryDirectory() as folder:
            config = self._write_config(folder)
            with self.assertRaisesRegex(TurboPhysAIError, "--accuracy"):
                _validate_run_groups(
                    config, ["demo.order"], [], ["python", "train.py"], False
                )
            _validate_run_groups(
                config, ["demo.order"], [], ["python", "train.py"], True
            )


if __name__ == "__main__":
    unittest.main()
