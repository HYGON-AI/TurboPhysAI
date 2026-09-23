# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from turbo_physai.engine.config import generator
from turbo_physai.engine.config.generation_record import (
    describe_input, file_hash, record_path, verify_generated, write_generated,
)
from turbo_physai.engine.config.loader import OptimizationConfigCatalog
from turbo_physai.engine.errors import OptimizationConfigError


ROOT = Path(__file__).resolve().parents[2]


class GenerationRecordTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "model"
        self.repo.mkdir()
        (self.repo / "tiny_model.py").write_text("def forward(x):\n    return x + 1\n")
        self.git("init", "-q")
        self.git("add", "tiny_model.py")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "baseline")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.project = self.root / "project"
        self.project.mkdir()
        self.catalog = self.project / "sample_catalog.py"
        self.catalog.write_text(
            "from turbo_physai import group, replace\n"
            "import tiny_model\n"
            "def optimized(x):\n    return x + 1\n"
            "FEATURE = group('sample.feature', replace(\n"
            "    target='tiny_model.forward', replacement='sample_catalog.optimized'))\n"
        )
        self.extra_catalog = self.project / "extra_catalog.py"
        self.extra_catalog.write_text("# Additional declared Catalog.\n")
        self.recipe = self.project / "configs" / "recipe.yaml"
        self.recipe.parent.mkdir()
        self.recipe.write_text(yaml.safe_dump({
            "metadata": {"id": "sample", "version": "1"},
            "model": {"name": "sample"},
            "optimization_modules": ["sample_catalog", "extra_catalog"],
            "optimization_groups": [{"id": "sample.feature", "options": {"mode": "default"}}],
        }))
        self.output = self.recipe.with_name("optimization.yaml")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True)

    def cli(self, *args):
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(
            (str(ROOT), str(self.project), str(self.repo))
        ))
        return subprocess.run(
            [sys.executable, "-m", "turbo_physai.cli", "optimization", *args],
            env=env, cwd=self.project, text=True, capture_output=True, timeout=30,
        )

    def generate(self, *extra):
        return self.cli(
            "generate", "--recipe", str(self.recipe), "--repo", str(self.repo),
            "--commit", self.commit, "--output", str(self.output), *extra,
        )

    def assert_generated(self):
        result = self.generate()
        self.assertEqual(result.returncode, 0, result.stderr)
        return verify_generated(self.output)

    def test_cli_generation_and_verification_without_model_environment(self):
        record = self.assert_generated()
        self.assertEqual(record["model_commit"], self.commit)
        self.assertEqual(record_path(self.output).name, ".optimization.yaml.generation.json")
        self.assertEqual([entry["role"] for entry in record["inputs"]], ["recipe", "catalog", "catalog"])
        self.assertNotIn(str(self.root), json.dumps(record))
        shutil.rmtree(self.repo)
        # The Catalog imports tiny_model, which no longer exists. Verification
        # must parse its source without importing it.
        result = self.cli("check", "--generated-only", str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        moved = self.root / "relocated"
        shutil.copytree(self.project, moved)
        shutil.rmtree(self.project)
        verify_generated(moved / "configs" / "optimization.yaml")

    def test_catalog_hash_option_is_recorded_but_not_emitted_to_config(self):
        self.catalog.write_text(self.catalog.read_text().replace(
            "replacement='sample_catalog.optimized'",
            "replacement='sample_catalog.optimized', collect_target_hash=False",
        ))
        self.assert_generated()
        generated = yaml.safe_load(self.output.read_text())
        entry = generated["optimization_groups"][0]
        self.assertNotIn("collect_target_hash", entry)
        self.assertEqual(entry["trust"], {"source_hashes": {}, "ast_hashes": {}})
        result = self.cli("check", "--generated-only", str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.cli("check", str(self.output), "--repo", str(self.repo))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.catalog.write_text(self.catalog.read_text().replace(
            "collect_target_hash=False", "collect_target_hash=True",
        ))
        with self.assertRaisesRegex(OptimizationConfigError, "file changed"):
            verify_generated(self.output)

    def test_changes_to_any_recorded_file_are_rejected(self):
        self.assert_generated()
        for path in (self.recipe, self.catalog, self.extra_catalog, self.output):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                try:
                    if path.suffix == ".py":
                        path.write_bytes(original + b"\nCHANGED = True\n")
                    else:
                        data = yaml.safe_load(original)
                        data["metadata"]["version"] = "2"
                        path.write_text(yaml.safe_dump(data))
                    with self.assertRaisesRegex(OptimizationConfigError, "file changed"):
                        verify_generated(self.output)
                finally:
                    path.write_bytes(original)

    def test_generated_header_follows_recipe_on_generation_and_overwrite(self):
        body = self.recipe.read_text()
        headers = (
            "# Copyright 2026 Hygon Information Technology Co., Ltd.\n"
            "# SPDX-License-Identifier: BSD-3-Clause\n\n",
            "# Copyright 2021-2027 Example contributors\n"
            "# Copyright 2024 Another contributor\n"
            "# SPDX-License-Identifier: MIT\n\n",
            "",
        )
        generated_body = None
        for header in headers:
            with self.subTest(header=header):
                self.recipe.write_text(header + body)
                result = self.generate("--force")
                self.assertEqual(result.returncode, 0, result.stderr)
                output = self.output.read_text()
                if generated_body is None:
                    generated_body = output[len(header):]
                    self.assertFalse(generated_body.startswith("#"))
                self.assertEqual(output, header + generated_body)
                verify_generated(self.output)

    def test_missing_or_invalid_record_and_missing_input_are_rejected(self):
        self.assert_generated()
        receipt = record_path(self.output)
        original = receipt.read_bytes()
        for content in (None, b"not json", b"{}"):
            with self.subTest(content=content):
                if content is None:
                    receipt.unlink()
                else:
                    receipt.write_bytes(content)
                with self.assertRaisesRegex(OptimizationConfigError, "regenerate"):
                    verify_generated(self.output)
                result = self.cli("check", "--generated-only", str(self.output))
                self.assertEqual(result.returncode, 2, result.stderr)
        receipt.write_bytes(original)
        self.catalog.unlink()
        with self.assertRaisesRegex(OptimizationConfigError, "verification failed"):
            verify_generated(self.output)

    def test_comments_and_formatting_do_not_invalidate_record(self):
        self.assert_generated()
        receipt = record_path(self.output).read_bytes()
        for path in (self.recipe, self.output):
            data = yaml.safe_load(path.read_text())
            data = dict(reversed(list(data.items())))
            formatted = yaml.safe_dump(data, sort_keys=False, default_flow_style=True, width=40)
            path.write_bytes(("# description\n\n" + formatted + "\n# end\n").replace("\n", "\r\n").encode())
        self.catalog.write_text(
            "# Catalog documentation\n\n" + self.catalog.read_text()
            .replace("    return x + 1", "        return (\n            x + 1\n        )  # unchanged")
            .replace("'tiny_model.forward'", '"tiny_model.forward"')
        )
        self.extra_catalog.write_text("# More documentation\n\n")
        shutil.rmtree(self.repo)
        result = self.cli("check", "--generated-only", str(self.output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(record_path(self.output).read_bytes(), receipt)

    def test_values_order_strings_and_python_logic_remain_significant(self):
        cases = (
            ("recipe", "value: 1\n", 'value: "1"\n'),
            ("config", "groups: [first, second]\n", "groups: [second, first]\n"),
            ("inherited_config", 'text: "a b"\n', 'text: "a  b"\n'),
            ("recipe", "text: |\n  a\n  b\n", "text: |\n  a b\n"),
            ("catalog", "target = 'model.forward'\n", "target = 'model.backward'\n"),
            ("catalog", "def f(x):\n    return x + 1\n", "def f(x):\n    return x + 2\n"),
            ("catalog", "if enabled:\n    first()\nsecond()\n", "if enabled:\n    first()\n    second()\n"),
            ("catalog", 'text = "a b"\n', 'text = "a  b"\n'),
        )
        path = self.root / "content"
        for role, before, after in cases:
            with self.subTest(role=role, before=before):
                path.write_text(before)
                original = file_hash(path, role)
                path.write_text(after)
                self.assertNotEqual(file_hash(path, role), original)

    def test_invalid_syntax_and_old_hash_scheme_are_rejected(self):
        self.assert_generated()
        for path, invalid in ((self.catalog, "def broken(:\n"), (self.output, "groups: [\n")):
            original = path.read_bytes()
            path.write_text(invalid)
            result = self.cli("check", "--generated-only", str(self.output))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("cannot parse", result.stderr)
            self.assertIn(path.name, result.stderr)
            self.assertIn("regenerate", result.stderr)
            path.write_bytes(original)
        receipt = record_path(self.output)
        record = json.loads(receipt.read_text())
        record["schema_version"] = "turbophysai/generation-record/v1"
        receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(OptimizationConfigError, "unsupported generation record"):
            verify_generated(self.output)

    def test_force_regenerates_pair_and_failed_generation_preserves_pair(self):
        self.assert_generated()
        before = (self.output.read_bytes(), record_path(self.output).read_bytes())
        self.assertNotEqual(self.generate().returncode, 0)
        self.catalog.write_text(self.catalog.read_text() + "\nUPDATED = True\n")
        result = self.generate("--force")
        self.assertEqual(result.returncode, 0, result.stderr)
        verify_generated(self.output)
        self.assertNotEqual(record_path(self.output).read_bytes(), before[1])
        before = (self.output.read_bytes(), record_path(self.output).read_bytes())
        (self.repo / "tiny_model.py").write_text("def forward(x):\n    return x + 2\n")
        result = self.generate("--force")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.output.read_bytes(), record_path(self.output).read_bytes()), before)

    def test_failed_initial_generation_does_not_create_receipt(self):
        (self.repo / "untracked").write_text("dirty")
        self.assertNotEqual(self.generate().returncode, 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(record_path(self.output).exists())

    def test_packaged_inputs_are_relative_to_installed_package(self):
        installed = self.root.resolve() / "installed"
        installed.mkdir()
        recipe = installed / "recipe.yaml"
        recipe.write_text("# packaged recipe\n")
        with patch("turbo_physai.engine.config.generation_record.PACKAGE_ROOT", installed):
            entries = [describe_input(recipe, "recipe", self.output)]
            self.assertEqual(entries[0]["base"], "package")
            self.assertEqual(entries[0]["path"], "recipe.yaml")
            write_generated(self.output, "# output\n", entries, self.commit)
        moved = self.root.resolve() / "moved_installation"
        installed.rename(moved)
        with patch("turbo_physai.engine.config.generation_record.PACKAGE_ROOT", moved):
            verify_generated(self.output)

    def test_inherited_configs_are_recorded_transitively(self):
        packaged = self.root / "packaged"
        inherited_paths = []
        for name, parents in (("base", []), ("framework", ["base"])):
            path = packaged / name / "configs" / "optimization.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump({
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": name, "version": "1"}, "extends": parents,
            }))
            inherited_paths.append(path)
        with patch("turbo_physai.engine.config.loader.PACKAGED_OPTIMIZATION_ROOT", packaged):
            catalog = OptimizationConfigCatalog.from_builtin_files()
            entries = generator._generation_inputs(
                self.recipe, SimpleNamespace(extends=("framework", "base")),
                SimpleNamespace(optimization_modules=()), catalog, self.output,
            )
        self.assertEqual(sum(entry["role"] == "inherited_config" for entry in entries), 2)
        write_generated(self.output, "# synthetic output\n", entries, self.commit)
        verify_generated(self.output)
        for path in inherited_paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"\n# documentation\n")
                verify_generated(self.output)
                path.write_bytes(original + b"\ndescription: changed\n")
                with self.assertRaisesRegex(OptimizationConfigError, "file changed"):
                    verify_generated(self.output)
                path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
