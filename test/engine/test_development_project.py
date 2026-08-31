# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

from turbo_physai.cli import main as cli_main
from turbo_physai.engine.errors import OptimizationConfigError
from turbo_physai.engine.config.loader import load_optimization_config


class DevelopmentProjectTest(unittest.TestCase):
    def test_cli_creates_blank_optimization_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bevformer-dev"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = cli_main(
                    [
                        "optimization",
                        "init",
                        "BEVFormer",
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue().strip(), str(output.resolve()))
            expected = {
                "README.md",
                "pyproject.toml",
                "bevformer_optimization/__init__.py",
                "bevformer_optimization/catalog.py",
                "bevformer_optimization/replacements.py",
                "configs/recipe.yaml",
                "tests/test_catalog.py",
            }
            actual = {
                str(path.relative_to(output))
                for path in output.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, expected)

            raw = yaml.safe_load(
                (output / "configs/recipe.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(raw["extends"], ["common.hcu.base"])
            self.assertEqual(
                raw["optimization_modules"],
                ["bevformer_optimization.catalog"],
            )
            self.assertEqual(raw["optimization_groups"], [])
            catalog_source = (
                output / "bevformer_optimization/catalog.py"
            ).read_text(encoding="utf-8")
            self.assertNotIn("target=", catalog_source)
            project_source = (output / "pyproject.toml").read_text(
                encoding="utf-8"
            )
            self.assertIn('requires-python = ">=3.10"', project_source)
            self.assertIn(
                'name = "turbo-physai-bevformer-optimization"',
                project_source,
            )
            readme_source = (output / "README.md").read_text(encoding="utf-8")
            self.assertIn("turbo-physai run", readme_source)
            self.assertNotIn("turbo_physai.apply", readme_source)

    def test_init_refuses_invalid_name_and_existing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = cli_main(
                    [
                        "optimization",
                        "init",
                        "model",
                        "--output",
                        str(existing),
                    ]
                )
            self.assertEqual(result, 2)
            self.assertIn("refusing to overwrite", stderr.getvalue())

            with self.assertRaisesRegex(
                OptimizationConfigError, "optimization name"
            ):
                from turbo_physai.development import create_optimization_project

                create_optimization_project("1invalid", root / "invalid")

    def test_model_optimization_config_imports_declared_optimization_module(self):
        with tempfile.TemporaryDirectory() as directory:
            optimization_config_path = Path(directory) / "config.yaml"
            optimization_config_path.write_text(
                """
schema_version: turbophysai/optimization-config/v1
kind: OptimizationConfig
metadata: {id: external, version: "1"}
optimization_modules:
  - customer_optimization.catalog
optimization_groups: []
""",
                encoding="utf-8",
            )
            with patch(
                "turbo_physai.engine.config.loader.importlib.import_module"
            ) as import_module:
                config = load_optimization_config(optimization_config_path)

            self.assertEqual(
                config.optimization_modules,
                ("customer_optimization.catalog",),
            )
            import_module.assert_called_once_with(
                "customer_optimization.catalog"
            )

    def test_schema_rejects_duplicate_optimization_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            optimization_config_path = Path(directory) / "config.yaml"
            optimization_config_path.write_text(
                """
schema_version: turbophysai/optimization-config/v1
kind: OptimizationConfig
metadata: {id: external, version: "1"}
optimization_modules:
  - customer_optimization.catalog
  - customer_optimization.catalog
optimization_groups: []
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                OptimizationConfigError, "must not contain duplicates"
            ):
                load_optimization_config(optimization_config_path)


if __name__ == "__main__":
    unittest.main()
