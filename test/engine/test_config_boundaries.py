# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration selection, evidence failures and interrupted generation."""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from turbo_physai.engine.config import generator, generation_record as records, loader
from turbo_physai.engine.config.schema import optimization_config_from_dict
from turbo_physai.engine.contracts import (
    OptimizationGroup,
    RestoreResult,
    RestoreStatus,
)
from turbo_physai.engine.definitions import (
    group,
    replace as declare_replace,
    replace_import,
)
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.errors import (
    OptimizationConfigError,
    OptimizationConfigNotFoundError,
)


def config(entries=(), **kwargs):
    return optimization_config_from_dict(
        dict(
            schema_version="turbophysai/optimization-config/v1",
            kind="OptimizationConfig",
            metadata={"id": "test", "version": "1"},
            optimization_groups=list(entries),
            **kwargs,
        )
    )


class ConfigBoundariesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_config(self, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        from turbo_physai.engine.config.schema import optimization_config_to_dict

        path.write_text(yaml.safe_dump(optimization_config_to_dict(config(**kwargs))))
        return path.resolve()

    def test_config_selection_precedence_and_model_names(self):
        explicit = self.write_config(self.root / "explicit.yaml")
        env_path = self.write_config(self.root / "env.yaml")
        model_path = self.write_config(
            self.root / "models" / "test_model" / "configs" / "optimization.yaml"
        )
        conventional = self.write_config(
            self.root / "turbophysai_configs" / "default" / "optimization.yaml"
        )
        fallback = self.write_config(self.root / "fallback.yaml")
        with patch.object(
            loader, "PACKAGED_MODEL_OPTIMIZATION_ROOT", self.root / "models"
        ), patch.object(
            loader, "PACKAGED_DEFAULT_OPTIMIZATION_CONFIG", fallback
        ), patch.object(
            Path, "cwd", return_value=self.root
        ), patch.dict(
            os.environ, {"TURBO_PHYSAI_OPTIMIZATION_CONFIG": str(env_path)}
        ):
            self.assertEqual(
                loader.resolve_optimization_config_path(explicit, model="missing"),
                explicit,
            )
            self.assertEqual(
                loader.resolve_optimization_config_path(model=" Test-Model "),
                model_path,
            )
            self.assertEqual(loader.resolve_optimization_config_path(), env_path)
            with patch.dict(os.environ, {"TURBO_PHYSAI_OPTIMIZATION_CONFIG": ""}):
                self.assertEqual(
                    loader.resolve_optimization_config_path(), conventional
                )
                conventional.unlink()
                self.assertEqual(loader.resolve_optimization_config_path(), fallback)
            for invalid in ("", "../test_model", "模型"):
                with self.subTest(invalid=invalid), self.assertRaises(
                    OptimizationConfigError
                ):
                    loader.resolve_optimization_config_path(model=invalid)
            with self.assertRaisesRegex(OptimizationConfigNotFoundError, "test_model"):
                loader.resolve_optimization_config_path(model="missing")
        with patch.object(
            loader, "PACKAGED_MODEL_OPTIMIZATION_ROOT", self.root / "absent"
        ):
            with self.assertRaisesRegex(
                OptimizationConfigNotFoundError, "available models: none"
            ):
                loader.resolve_optimization_config_path(model="missing")

    def test_overlay_merges_trust_by_target_without_losing_base_options(self):
        base = config(
            [
                {
                    "id": "g",
                    "enabled": False,
                    "options": {"a": 1},
                    "trust": {
                        "source_hashes": {"a.target": ["old"], "b.target": ["keep"]}
                    },
                }
            ]
        )
        overlay = config(
            [
                {
                    "id": "g",
                    "enabled": True,
                    "trust": {
                        "source_hashes": {"a.target": ["new"]},
                        "ast_hashes": {"a.target": ["syntax"]},
                    },
                }
            ]
        )
        merged = loader._merge(base, overlay).optimization_groups[0]
        self.assertTrue(merged.enabled)
        self.assertEqual(dict(merged.options), {"a": 1})
        self.assertEqual(merged.trust["source_hashes"]["b.target"], ("keep",))
        self.assertEqual(merged.trust["source_hashes"]["a.target"], ("new",))
        self.assertEqual(merged.trust["ast_hashes"]["a.target"], ("syntax",))
        broken = config(optimization_modules=["missing_catalog_for_boundary_test"])
        with self.assertRaisesRegex(
            OptimizationConfigError, "failed to import optimization module"
        ):
            loader.resolve_optimization_config(
                broken, catalog=loader.OptimizationConfigCatalog()
            )
        self.assertEqual(
            loader.resolve_optimization_config(
                broken, import_modules=False
            ).optimization_modules,
            broken.optimization_modules,
        )

    def test_generation_record_rejects_malformed_metadata(self):
        recipe = self.root / "recipe.yaml"
        recipe.write_text("a: 1\n")
        output = self.root / "optimization.yaml"
        inputs = [records.describe_input(recipe, "recipe", output)]
        records.write_generated(output, "x: 1\n", inputs, "a" * 40)
        receipt = records.record_path(output)
        original = json.loads(receipt.read_text())
        mutations = [
            lambda r: r.update(model_commit=123),
            lambda r: r.update(model_commit="bad"),
            lambda r: r["config"].update(path="other.yaml"),
            lambda r: r.update(inputs=[]),
            lambda r: r["inputs"][0].update(role="catalog"),
            lambda r: r["inputs"][0].update(role="unknown"),
            lambda r: r["inputs"][0].update(base="unknown"),
            lambda r: r["inputs"][0].update(path="/absolute/file"),
            lambda r: r["config"].update(sha256="broken"),
            lambda r: r["inputs"][0].update(sha256=None),
        ]
        for mutate in mutations:
            record = json.loads(json.dumps(original))
            mutate(record)
            receipt.write_text(json.dumps(record))
            with self.subTest(record=record), self.assertRaisesRegex(
                OptimizationConfigError, "verification failed"
            ):
                records.verify_generated(output)
        receipt.write_text(json.dumps(original))
        self.assertEqual(records.verify_generated(output), original)
        with self.assertRaisesRegex(OptimizationConfigError, "overwrite"):
            records.write_generated(output, "x: 2", inputs, "a" * 40)
        recipe.write_text("a: 2")
        with self.assertRaisesRegex(OptimizationConfigError, "inputs changed"):
            records.write_generated(output, "x: 2", inputs, "a" * 40, force=True)
        self.assertEqual(output.read_text(), "x: 1\n")

    def test_interrupted_publication_is_detected_and_temp_files_are_removed(self):
        recipe = self.root / "recipe.yaml"
        recipe.write_text("a: 1")
        output = self.root / "optimization.yaml"
        inputs = [records.describe_input(recipe, "recipe", output)]
        records.write_generated(output, "x: 1", inputs, "a" * 40)
        original_replace = os.replace

        def fail_receipt(source, destination):
            if destination == records.record_path(output):
                raise OSError("disk full")
            original_replace(source, destination)

        with patch.object(records.os, "replace", side_effect=fail_receipt):
            with self.assertRaisesRegex(OSError, "disk full"):
                records.write_generated(output, "x: 2", inputs, "a" * 40, force=True)
        self.assertEqual(
            {p.name for p in self.root.iterdir()},
            {"recipe.yaml", "optimization.yaml", ".optimization.yaml.generation.json"},
        )
        with self.assertRaisesRegex(OptimizationConfigError, "file changed"):
            records.verify_generated(output)

    def test_generate_to_file_checks_checkout_and_publishes_receipt(self):
        # No model dependency is needed for a recipe with no selected groups.
        import subprocess

        repo = self.root / "repo"
        repo.mkdir()

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(repo), *args], text=True
            ).strip()

        git("init", "-q")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "initial",
        )
        commit = git("rev-parse", "HEAD")
        recipe = self.root / "recipe.yaml"
        recipe.write_text(
            '# preserved notice\nmetadata: {id: test, version: "1"}\noptimization_groups: []\n'
        )
        output = self.root / "generated.yaml"
        generator.generate_to_file(recipe, repo, commit, output)
        self.assertEqual(records.verify_generated(output)["model_commit"], commit)
        self.assertTrue(output.read_text().startswith("# preserved notice\n"))
        self.assertEqual(
            generator.check_optimization_config(output, repo).optimization_groups, ()
        )
        with self.assertRaisesRegex(OptimizationConfigError, "overwrite"):
            generator.generate_to_file(recipe, repo, commit, output)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = generator.main(
                [
                    "--recipe",
                    str(recipe),
                    "--repo",
                    str(repo),
                    "--commit",
                    commit,
                    "--output",
                    str(output),
                    "--force",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().strip(), str(output))
        recipe.write_text("[]")
        with self.assertRaisesRegex(OptimizationConfigError, "must be a mapping"):
            generator.generate(recipe, repo, commit)
        with self.assertRaises(OptimizationConfigError):
            generator._git(self.root, "rev-parse", "HEAD")

    def test_evidence_failure_restores_working_directory_and_import_path(self):
        registry = Registry()
        group(
            "missing.target",
            declare_replace("boundary_missing.fn", "unused.fn"),
            registry=registry,
        )
        registry.register_group(OptimizationGroup("missing.spec", ("no.such.spec",)))
        cwd, path = Path.cwd(), list(sys.path)
        with patch.object(generator, "default_registry", registry):
            for group_id, message in [
                ("absent", "Group is not registered"),
                ("missing.spec", "ReplacementSpec is not registered"),
                ("missing.target", "cannot resolve evidence"),
            ]:
                with self.subTest(group=group_id), self.assertRaisesRegex(
                    OptimizationConfigError, message
                ):
                    generator._collect_group_evidence(
                        config([{"id": group_id}]), self.root
                    )
                self.assertEqual(Path.cwd(), cwd)
                self.assertEqual(sys.path, path)
            with self.assertRaisesRegex(OptimizationConfigError, "not registered"):
                generator._validate_group_composition(config([{"id": "absent"}]))
            registry.register_group(
                OptimizationGroup("a", ("unused",), depends_on=("b",))
            )
            registry.register_group(
                OptimizationGroup("b", ("unused",), depends_on=("a",))
            )
            with self.assertRaisesRegex(OptimizationConfigError, "dependency cycle"):
                generator._expand_group_dependencies(config([{"id": "a"}]))
            with self.assertRaisesRegex(OptimizationConfigError, "disabled Group"):
                generator._expand_group_dependencies(
                    config([{"id": "a"}, {"id": "b", "enabled": False}])
                )

    def test_temporary_imports_rollback_on_apply_and_restore_failure(self):
        registry = Registry()
        group(
            "imports",
            replace_import("boundary_import", "boundary_replacement"),
            registry=registry,
        )
        selected = config([{"id": "imports"}])
        from turbo_physai.engine.execution.replacements.import_replace import (
            ImportReplaceHandler,
        )
        from turbo_physai.engine.contracts import Mechanism

        replacement = types.ModuleType("boundary_replacement")
        with patch.dict(
            sys.modules, {"boundary_replacement": replacement}
        ), patch.object(generator, "default_registry", registry):
            with generator._temporary_import_compatibility(selected):
                self.assertIs(sys.modules["boundary_import"], replacement)
            self.assertNotIn("boundary_import", sys.modules)
            handler = ImportReplaceHandler()
            apply = handler.apply

            def failed_apply(prepared):
                apply(prepared)
                raise RuntimeError("apply failed")

            with patch.object(
                generator,
                "default_handlers",
                return_value={Mechanism.IMPORT_REPLACE: handler},
            ), patch.object(handler, "apply", side_effect=failed_apply):
                with self.assertRaisesRegex(OptimizationConfigError, "apply failed"):
                    with generator._temporary_import_compatibility(selected):
                        self.fail("must not enter context")
                self.assertNotIn("boundary_import", sys.modules)
            restore = handler.restore

            def failed_restore(snapshot):
                restore(snapshot)
                return (
                    RestoreResult(
                        "boundary_import", RestoreStatus.FAILED, "restore failed"
                    ),
                )

            with patch.object(
                generator,
                "default_handlers",
                return_value={Mechanism.IMPORT_REPLACE: handler},
            ), patch.object(handler, "restore", side_effect=failed_restore):
                with self.assertRaisesRegex(OptimizationConfigError, "restore failed"):
                    with generator._temporary_import_compatibility(selected):
                        pass
