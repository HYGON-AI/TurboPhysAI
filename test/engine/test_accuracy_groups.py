"""accuracy_only Group 开关：精度验证专用 Group 只在该模式下应用。"""

import unittest

from turbo_physai.bootstrap import ACCURACY_MODE, bootstrap_environment
from turbo_physai.engine import _apply_group_overrides
from turbo_physai.engine.config.loader import _merge
from turbo_physai.engine.config.schema import optimization_config_from_dict
from turbo_physai.engine.contracts import (
    OptimizationConfig,
    OptimizationConfigMetadata,
    OptimizationGroupConfig,
)
from turbo_physai.engine.errors import OptimizationConfigError


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


if __name__ == "__main__":
    unittest.main()
