# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from turbo_physai.engine.errors import OptimizationConfigError
from turbo_physai.engine.contracts import (
    FrozenDict,
    Mechanism,
    ReplacementSpec,
)
from turbo_physai.engine.config.loader import OptimizationConfigCatalog, load_optimization_config
from turbo_physai.engine.config.schema import optimization_config_from_dict


def optimization_config_text(group="demo.group", extra=""):
    return textwrap.dedent(
        f"""
        schema_version: turbophysai/optimization-config/v1
        kind: OptimizationConfig
        metadata:
          id: demo
          version: "1.0"
        model:
          name: demo
        compatibility: {{}}
        optimization_groups:
          - id: {group}
        {extra}
        """
    )


class ModelsAndConfigsTest(unittest.TestCase):
    def test_import_does_not_load_torch(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import turbo_physai; assert 'torch' not in sys.modules",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_engine_exports_only_developer_declarations(self):
        from turbo_physai import engine

        self.assertTrue(callable(engine.group))
        self.assertTrue(callable(engine.replace))
        self.assertTrue(callable(engine.replace_import))
        self.assertTrue(callable(engine.wrap))
        self.assertFalse(hasattr(engine, "ReplacementSpec"))
        self.assertFalse(hasattr(engine, "default_registry"))

    def test_replacement_spec_keeps_declared_paths(self):
        spec = ReplacementSpec(
            "x",
            Mechanism.REPLACE,
            "m.f",
            "r",
        )
        self.assertEqual(spec.target, "m.f")
        self.assertEqual(spec.replacement, "r")

    def test_frozen_options(self):
        value = FrozenDict({"nested": {"items": [1, 2]}})
        with self.assertRaises(TypeError):
            value["x"] = 1
        self.assertEqual(value["nested"]["items"], (1, 2))

    def test_optimization_config_load_ignores_yaml_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.yaml"
            second = Path(directory) / "second.yaml"
            first.write_text(optimization_config_text(), encoding="utf-8")
            second.write_text("# comment\n" + optimization_config_text(), encoding="utf-8")
            self.assertEqual(load_optimization_config(first), load_optimization_config(second))

    def test_unknown_field_rejected(self):
        raw = {
            "schema_version": "turbophysai/optimization-config/v1",
            "kind": "OptimizationConfig",
            "metadata": {"id": "demo", "version": "1"},
            "unexpected": True,
        }
        with self.assertRaises(OptimizationConfigError):
            optimization_config_from_dict(raw)

    def test_invalid_compatibility_shape_rejected(self):
        raw = {
            "schema_version": "turbophysai/optimization-config/v1",
            "kind": "OptimizationConfig",
            "metadata": {"id": "demo", "version": "1"},
            "compatibility": {"dependencies": {"torch": 123}},
        }
        with self.assertRaises(OptimizationConfigError):
            optimization_config_from_dict(raw)
        raw["compatibility"] = {"dependencies": {"torch": ">=not-a-version"}}
        with self.assertRaises(OptimizationConfigError):
            optimization_config_from_dict(raw)

    def test_optimization_config_commit_constraint_is_supported_as_metadata(self):
        raw = {
            "schema_version": "turbophysai/optimization-config/v1",
            "kind": "OptimizationConfig",
            "metadata": {"id": "demo", "version": "1"},
            "compatibility": {"commits": ["abc123"]},
        }
        config = optimization_config_from_dict(raw)
        self.assertEqual(config.compatibility["commits"], ("abc123",))

    def test_catalog_inheritance_merges_group_by_id(self):
        parent = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "framework.base", "version": "1"},
                "optimization_groups": [
                    {
                        "id": "demo.group",
                        "options": {"a": 1, "retained": True},
                    }
                ],
            }
        )
        catalog = OptimizationConfigCatalog({"framework.base": parent})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: child, version: "1"}
                    extends: [framework.base]
                    optimization_groups:
                      - id: demo.group
                        options: {a: 2}
                    """
                ),
                encoding="utf-8",
            )
            config = load_optimization_config(path, catalog=catalog)
        self.assertEqual(len(config.optimization_groups), 1)
        self.assertEqual(config.optimization_groups[0].options["a"], 2)
        self.assertTrue(config.optimization_groups[0].options["retained"])

    def test_builtin_model_optimization_config_is_flattened_after_generation(self):
        catalog = OptimizationConfigCatalog.from_builtin_files()
        common = catalog.get("common.hcu.base")
        bevformer = catalog.get("model.bevformer.base.hcu")
        self.assertIsNotNone(common)
        self.assertEqual(common.optimization_groups, ())
        self.assertEqual(bevformer.extends, ())
        self.assertEqual(len(bevformer.optimization_groups), 7)

    def test_environment_optimization_config_path_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(optimization_config_text(), encoding="utf-8")
            previous = os.environ.get("TURBO_PHYSAI_OPTIMIZATION_CONFIG")
            os.environ["TURBO_PHYSAI_OPTIMIZATION_CONFIG"] = str(path)
            try:
                self.assertEqual(load_optimization_config().metadata.id, "demo")
            finally:
                if previous is None:
                    os.environ.pop("TURBO_PHYSAI_OPTIMIZATION_CONFIG", None)
                else:
                    os.environ["TURBO_PHYSAI_OPTIMIZATION_CONFIG"] = previous


if __name__ == "__main__":
    unittest.main()
