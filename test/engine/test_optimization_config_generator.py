# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import io
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from turbo_physai.engine.errors import OptimizationConfigError
from turbo_physai.engine.contracts import Mechanism, OptimizationGroup, ReplacementSpec
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.config import generator
from turbo_physai.engine.config.loader import OptimizationConfigCatalog, resolve_optimization_config
from turbo_physai.engine.config.schema import optimization_config_from_dict


TEMPLATE = textwrap.dedent(
    """
    metadata: {id: generated, version: "1"}
    model: {name: bevformer}
    optimization_modules:
      - turbo_physai.optimizations.models.bevformer.catalog
    extends: [common.hcu.base]
    optimization_groups:
      - id: bevformer.msda
    """
)


class OptimizationConfigGeneratorTest(unittest.TestCase):
    def test_import_replace_does_not_generate_replacement_hash_evidence(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "import.member",
                Mechanism.IMPORT_REPLACE,
                "missing_upstream.module",
                "turbo_physai.engine.config.generator",
            )
        )
        registry.register_group(
            OptimizationGroup("import.group", ("import.member",))
        )
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [{"id": "import.group"}],
            }
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(
            generator, "default_registry", registry
        ), patch.object(generator, "source_hash") as source, patch.object(
            generator, "ast_hash"
        ) as syntax:
            evidence = generator._collect_group_evidence(config, Path(directory))

        self.assertEqual(
            evidence["import.group"],
            {"source_hashes": {}, "ast_hashes": {}},
        )
        source.assert_not_called()
        syntax.assert_not_called()

    def test_dependency_closure_is_materialized_from_catalog(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "base.member",
                Mechanism.REPLACE,
                "base.target",
                "base.replacement",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model.target",
                "model.replacement",
            )
        )
        registry.register_group(OptimizationGroup("base.group", ("base.member",)))
        registry.register_group(
            OptimizationGroup(
                "model.group",
                ("model.member",),
                depends_on=("base.group",),
            )
        )
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [{"id": "model.group"}],
            }
        )
        with patch.object(generator, "default_registry", registry):
            expanded = generator._expand_group_dependencies(config)
        self.assertEqual(
            tuple(entry.id for entry in expanded.optimization_groups),
            ("base.group", "model.group"),
        )

    def test_dependency_closure_rejects_explicitly_disabled_dependency(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "base.member",
                Mechanism.REPLACE,
                "base.target",
                "base.replacement",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model.target",
                "model.replacement",
            )
        )
        registry.register_group(OptimizationGroup("base.group", ("base.member",)))
        registry.register_group(
            OptimizationGroup(
                "model.group",
                ("model.member",),
                depends_on=("base.group",),
            )
        )
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [
                    {"id": "base.group", "enabled": False},
                    {"id": "model.group"},
                ],
            }
        )
        with patch.object(generator, "default_registry", registry):
            with self.assertRaisesRegex(OptimizationConfigError, "disabled Group"):
                generator._expand_group_dependencies(config)

    def test_dependency_closure_rejects_cycle(self):
        registry = Registry()
        for name in ("first", "second"):
            registry.register_spec(
                ReplacementSpec(
                    f"{name}.member",
                    Mechanism.REPLACE,
                    f"{name}.target",
                    f"{name}.replacement",
                )
            )
        registry.register_group(
            OptimizationGroup(
                "first.group",
                ("first.member",),
                depends_on=("second.group",),
            )
        )
        registry.register_group(
            OptimizationGroup(
                "second.group",
                ("second.member",),
                depends_on=("first.group",),
            )
        )
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [{"id": "first.group"}],
            }
        )
        with patch.object(generator, "default_registry", registry):
            with self.assertRaisesRegex(OptimizationConfigError, "dependency cycle"):
                generator._expand_group_dependencies(config)

    def test_generate_stores_advisory_commit_and_adds_hash_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            template.write_text(TEMPLATE, encoding="utf-8")
            previous_cwd = Path.cwd()

            with patch.object(
                generator, "_git", side_effect=["abc123", ""]
            ), patch.object(
                generator,
                "resolve_attribute",
                return_value=SimpleNamespace(original=object()),
            ), patch.object(
                generator, "source_hash", return_value="source-digest"
            ), patch.object(
                generator, "ast_hash", return_value="ast-digest"
            ), patch.object(
                generator, "_validate_public_replacement_references"
            ) as validate_references:
                rendered = generator.generate(template, root, "abc123")

            self.assertEqual(Path.cwd(), previous_cwd)
            raw = yaml.safe_load(rendered)
            self.assertEqual(
                raw["schema_version"], "turbophysai/optimization-config/v1"
            )
            self.assertEqual(raw["kind"], "OptimizationConfig")
            self.assertNotIn("extends", raw)
            self.assertEqual(raw["compatibility"]["commits"], ["abc123"])
            trust = raw["optimization_groups"][0]["trust"]
            self.assertEqual(len(trust["source_hashes"]), 2)
            self.assertEqual(len(trust["ast_hashes"]), 2)
            self.assertNotIn("# digest:", rendered)
            self.assertNotIn("config-v1:", rendered)
            validate_references.assert_called_once()

    def test_generate_rejects_group_conflicts_before_hash_collection(self):
        conflict_template = textwrap.dedent(
            """
            schema_version: turbophysai/optimization-config/v1
            kind: OptimizationConfig
            metadata: {id: generated, version: "1"}
            extends: [public.base]
            optimization_groups:
              - id: model.group
            """
        )
        public_config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "public.base", "version": "1"},
                "optimization_groups": [
                    {"id": "public.group"}
                ],
            }
        )
        catalog = OptimizationConfigCatalog({"public.base": public_config})
        for model_replacement, code in (
            ("model.replacement", "target.group_conflict"),
            ("public.replacement", "target.group_duplicate"),
        ):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                registry = Registry()
                registry.register_spec(
                    ReplacementSpec(
                        "public.member",
                        Mechanism.REPLACE,
                        "shared.target",
                        "public.replacement",
                    )
                )
                registry.register_spec(
                    ReplacementSpec(
                        "model.member",
                        Mechanism.REPLACE,
                        "shared.target",
                        model_replacement,
                    )
                )
                registry.register_group(
                    OptimizationGroup("public.group", ("public.member",))
                )
                registry.register_group(
                    OptimizationGroup("model.group", ("model.member",))
                )
                root = Path(directory)
                template = root / "template.yaml"
                template.write_text(conflict_template, encoding="utf-8")
                with patch.object(
                    generator, "_git", side_effect=["abc123", ""]
                ), patch.object(
                    generator, "default_registry", registry
                ), patch.object(
                    generator,
                    "resolve_optimization_config",
                    side_effect=lambda config, **_kwargs: resolve_optimization_config(
                        config, catalog=catalog
                    ),
                ), patch.object(
                    generator, "resolve_attribute"
                ) as resolve:
                    with self.assertRaisesRegex(OptimizationConfigError, code):
                        generator.generate(template, root, "abc123")
                resolve.assert_not_called()

    def test_public_reference_check_rejects_model_replacement(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "public.member",
                Mechanism.REPLACE,
                "standard_api.op",
                "dd_public_impl.optimized_op",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model_api.forward",
                "dd_model_impl.optimized_forward",
            )
        )
        registry.register_group(OptimizationGroup("public.group", ("public.member",)))
        registry.register_group(OptimizationGroup("model.group", ("model.member",)))
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [
                    {"id": "public.group"},
                    {"id": "model.group"},
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dd_public_impl.py").write_text(
                "def optimized_op(value):\n    return value\n",
                encoding="utf-8",
            )
            (root / "dd_model_impl.py").write_text(
                textwrap.dedent(
                    """
                    from dd_public_impl import optimized_op as fast_op

                    def optimized_forward(value):
                        return fast_op(value)
                    """
                ),
                encoding="utf-8",
            )
            try:
                with patch.object(generator, "default_registry", registry):
                    with self.assertRaisesRegex(
                        OptimizationConfigError,
                        "public.replacement_reference",
                    ) as raised:
                        generator._validate_public_replacement_references(
                            config, ("public.group",), root
                        )
                message = str(raised.exception)
                self.assertIn("model_group=model.group", message)
                self.assertIn("public_group=public.group", message)
                self.assertIn("use_standard_target=standard_api.op", message)
                self.assertIn("line=", message)
            finally:
                sys.modules.pop("dd_public_impl", None)
                sys.modules.pop("dd_model_impl", None)

    def test_public_reference_check_allows_calling_standard_target(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "public.member",
                Mechanism.REPLACE,
                "standard_api.op",
                "dd_public_impl.optimized_op",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model_api.forward",
                "dd_model_impl.optimized_forward",
            )
        )
        registry.register_group(OptimizationGroup("public.group", ("public.member",)))
        registry.register_group(OptimizationGroup("model.group", ("model.member",)))
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [
                    {"id": "public.group"},
                    {"id": "model.group"},
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dd_public_impl.py").write_text(
                "def optimized_op(value):\n    return value\n",
                encoding="utf-8",
            )
            (root / "standard_api.py").write_text(
                "def op(value):\n    return value\n",
                encoding="utf-8",
            )
            (root / "dd_model_impl.py").write_text(
                textwrap.dedent(
                    """
                    import standard_api

                    def optimized_forward(value):
                        return standard_api.op(value)
                    """
                ),
                encoding="utf-8",
            )
            try:
                with patch.object(generator, "default_registry", registry):
                    generator._validate_public_replacement_references(
                        config, ("public.group",), root
                    )
            finally:
                for module_name in (
                    "dd_public_impl",
                    "dd_model_impl",
                    "standard_api",
                ):
                    sys.modules.pop(module_name, None)

    def test_public_reference_check_detects_saved_module_alias(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "public.member",
                Mechanism.REPLACE,
                "standard_api.op",
                "dd_public_impl.optimized_op",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model_api.forward",
                "dd_model_impl.optimized_forward",
            )
        )
        registry.register_group(OptimizationGroup("public.group", ("public.member",)))
        registry.register_group(OptimizationGroup("model.group", ("model.member",)))
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [
                    {"id": "public.group"},
                    {"id": "model.group"},
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dd_public_impl.py").write_text(
                "def optimized_op(value):\n    return value\n",
                encoding="utf-8",
            )
            (root / "dd_model_impl.py").write_text(
                textwrap.dedent(
                    """
                    import dd_public_impl as public_impl

                    saved_op = public_impl.optimized_op

                    def optimized_forward(value):
                        return saved_op(value)
                    """
                ),
                encoding="utf-8",
            )
            try:
                with patch.object(generator, "default_registry", registry):
                    with self.assertRaisesRegex(
                        OptimizationConfigError,
                        "public.replacement_reference",
                    ):
                        generator._validate_public_replacement_references(
                            config, ("public.group",), root
                        )
            finally:
                sys.modules.pop("dd_public_impl", None)
                sys.modules.pop("dd_model_impl", None)

    def test_public_reference_check_detects_dynamic_import(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "public.member",
                Mechanism.REPLACE,
                "standard_api.op",
                "dd_public_impl.optimized_op",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model_api.forward",
                "dd_model_impl.optimized_forward",
            )
        )
        registry.register_group(OptimizationGroup("public.group", ("public.member",)))
        registry.register_group(OptimizationGroup("model.group", ("model.member",)))
        config = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "generated", "version": "1"},
                "optimization_groups": [
                    {"id": "public.group"},
                    {"id": "model.group"},
                ],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dd_public_impl.py").write_text(
                "def optimized_op(value):\n    return value\n",
                encoding="utf-8",
            )
            (root / "dd_model_impl.py").write_text(
                textwrap.dedent(
                    """
                    import importlib

                    def optimized_forward(value):
                        module = importlib.import_module("dd_public_impl")
                        return module.optimized_op(value)
                    """
                ),
                encoding="utf-8",
            )
            try:
                with patch.object(generator, "default_registry", registry):
                    with self.assertRaisesRegex(
                        OptimizationConfigError,
                        "public.replacement_reference",
                    ):
                        generator._validate_public_replacement_references(
                            config, ("public.group",), root
                        )
            finally:
                sys.modules.pop("dd_public_impl", None)
                sys.modules.pop("dd_model_impl", None)

    def test_generate_rejects_head_mismatch_and_dirty_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            template.write_text(TEMPLATE, encoding="utf-8")
            with patch.object(generator, "_git", return_value="different"):
                with self.assertRaisesRegex(OptimizationConfigError, "HEAD mismatch"):
                    generator.generate(template, root, "abc123")
            with patch.object(
                generator, "_git", side_effect=["abc123", " M model.py"]
            ):
                with self.assertRaisesRegex(OptimizationConfigError, "clean model"):
                    generator.generate(template, root, "abc123")

    def test_check_optimization_config_validates_generated_yaml_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            optimization_config_path = root / "generated.yaml"
            template.write_text(TEMPLATE, encoding="utf-8")
            with patch.object(
                generator, "_git", side_effect=["abc123", ""]
            ), patch.object(
                generator,
                "resolve_attribute",
                return_value=SimpleNamespace(original=object()),
            ), patch.object(
                generator, "source_hash", return_value="source-digest"
            ), patch.object(generator, "ast_hash", return_value="ast-digest"):
                rendered = generator.generate(template, root, "abc123")
            optimization_config_path.write_text(rendered, encoding="utf-8")

            with patch.object(generator, "_git", return_value=""), patch.object(
                generator,
                "resolve_attribute",
                return_value=SimpleNamespace(original=object()),
            ), patch.object(
                generator, "source_hash", return_value="source-digest"
            ), patch.object(generator, "ast_hash", return_value="ast-digest"):
                checked = generator.check_optimization_config(optimization_config_path, root)

            self.assertEqual(checked.metadata.id, "generated")
            self.assertEqual(optimization_config_path.read_text(encoding="utf-8"), rendered)

    def test_check_optimization_config_allows_commit_change_but_rejects_dirty_and_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "template.yaml"
            optimization_config_path = root / "generated.yaml"
            template.write_text(TEMPLATE, encoding="utf-8")
            with patch.object(
                generator, "_git", side_effect=["abc123", ""]
            ), patch.object(
                generator,
                "resolve_attribute",
                return_value=SimpleNamespace(original=object()),
            ), patch.object(
                generator, "source_hash", return_value="source-digest"
            ), patch.object(generator, "ast_hash", return_value="ast-digest"):
                optimization_config_path.write_text(
                    generator.generate(template, root, "abc123"),
                    encoding="utf-8",
                )

            with patch.object(generator, "_git", return_value=""), patch.object(
                generator,
                "resolve_attribute",
                return_value=SimpleNamespace(original=object()),
            ), patch.object(
                generator, "source_hash", return_value="source-digest"
            ), patch.object(generator, "ast_hash", return_value="ast-digest"):
                checked = generator.check_optimization_config(optimization_config_path, root)
            self.assertEqual(checked.metadata.id, "generated")

            with patch.object(generator, "_git", return_value=" M model.py"):
                with self.assertRaisesRegex(OptimizationConfigError, "clean model"):
                    generator.check_optimization_config(optimization_config_path, root)

            with patch.object(generator, "_git", return_value=""), patch.object(
                generator,
                "resolve_attribute",
                return_value=SimpleNamespace(original=object()),
            ), patch.object(
                generator, "source_hash", return_value="changed-source"
            ), patch.object(generator, "ast_hash", return_value="changed-ast"):
                with self.assertRaisesRegex(
                    OptimizationConfigError, "target evidence mismatch"
                ):
                    generator.check_optimization_config(optimization_config_path, root)

    def test_check_optimization_config_requires_generated_dependency_closure(self):
        registry = Registry()
        registry.register_spec(
            ReplacementSpec(
                "base.member",
                Mechanism.REPLACE,
                "base.target",
                "base.replacement",
            )
        )
        registry.register_spec(
            ReplacementSpec(
                "model.member",
                Mechanism.REPLACE,
                "model.target",
                "model.replacement",
            )
        )
        registry.register_group(OptimizationGroup("base.group", ("base.member",)))
        registry.register_group(
            OptimizationGroup(
                "model.group",
                ("model.member",),
                depends_on=("base.group",),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "incomplete.yaml"
            optimization_config_path.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: generated, version: "1"}
                    compatibility: {}
                    optimization_groups:
                      - id: model.group
                    """
                ),
                encoding="utf-8",
            )
            with patch.object(
                generator, "_git", return_value=""
            ), patch.object(generator, "default_registry", registry):
                with self.assertRaisesRegex(
                    OptimizationConfigError, "dependency closure/order mismatch"
                ):
                    generator.check_optimization_config(optimization_config_path, root)

    def test_main_reports_validation_error(self):
        stderr = io.StringIO()
        with patch.object(
            generator,
            "generate",
            side_effect=OptimizationConfigError("invalid checkout"),
        ), redirect_stderr(stderr):
            result = generator.main(
                [
                    "--recipe",
                    "template.yaml",
                    "--repo",
                    "repo",
                    "--commit",
                    "abc123",
                ]
            )
        self.assertEqual(result, 2)
        self.assertIn("invalid checkout", stderr.getvalue())

    def test_main_writes_generated_config(self):
        stdout = io.StringIO()
        with patch.object(generator, "generate", return_value="generated\n"), \
                redirect_stdout(stdout):
            result = generator.main(
                [
                    "--recipe",
                    "template.yaml",
                    "--repo",
                    "repo",
                    "--commit",
                    "abc123",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue(), "generated\n")


if __name__ == "__main__":
    unittest.main()
