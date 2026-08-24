# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import turbo_physai
import turbo_physai.engine as engine_module
from turbo_physai.engine.checking import checker as checker_module
from turbo_physai.engine.checking.checker import Checker
from turbo_physai.engine.checking.context import _int_env, _run_git, detect_context
from turbo_physai.engine.errors import (
    OptimizationConfigNotFoundError,
    OptimizationConfigError,
    ReportWriteError,
)
from turbo_physai.engine.contracts import (
    CheckResult,
    CheckStatus,
    EnvironmentSnapshot,
    FrozenDict,
    Mechanism,
    OptimizationConfig,
    OptimizationGroup,
    ReplacementSpec,
    OptimizationGroupConfig,
    OptimizationConfigMetadata,
    freeze_json,
    to_primitive,
)
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.execution.replacements import (
    HandlerError,
    MechanismHandler,
    default_handlers,
)
from turbo_physai.engine.execution.replacements.base import resolve_attribute, set_attribute
from turbo_physai.engine.execution.replacements.import_replace import ImportReplaceHandler
from turbo_physai.engine.config.loader import (
    OptimizationConfigCatalog,
    load_optimization_config,
    resolve_optimization_config_path,
)
from turbo_physai.engine.config.schema import optimization_config_from_dict


def target(value, extra=0):
    return value + extra


def incompatible(value):
    return value


def permissive(*args, **kwargs):
    return args


def adds_required_parameter(value, required):
    return value + required


def keyword_only_extra(value, *, extra=0):
    return value + extra


def all_argument_kinds(positional_only, /, positional=0, *args, keyword=1, **kwargs):
    return positional_only + positional + keyword


class NegativePathsTest(unittest.TestCase):
    def setUp(self):
        engine_module._apply_called = False

    def test_model_validation_and_serialization_edges(self):
        self.assertEqual(repr(FrozenDict({"a": 1})), "FrozenDict({'a': 1})")
        self.assertEqual(FrozenDict({"a": 1}), {"a": 1})
        self.assertNotEqual(FrozenDict({"a": 1}), object())
        self.assertEqual(freeze_json([{"a": 1}]), (FrozenDict({"a": 1}),))
        self.assertEqual(to_primitive((1, 2)), [1, 2])
        with self.assertRaises(ValueError):
            ReplacementSpec(
                "duplicate.alias",
                Mechanism.WRAPPER,
                "module.func",
                "replacement",
                aliases=("module.func",),
            )
        with self.assertRaises(ValueError):
            OptimizationGroup("duplicate.members", ("a", "a"))
        with self.assertRaises(ValueError):
            OptimizationGroup("self.dependency", ("a",), depends_on=("self.dependency",))
        with self.assertRaises(ValueError):
            OptimizationGroup("string.dependency", ("a",), depends_on="not-a-tuple")
        with self.assertRaises(ValueError):
            OptimizationConfigMetadata("", "1")
        with self.assertRaises(ValueError):
            OptimizationGroupConfig("")
        with self.assertRaises(ValueError):
            OptimizationConfig("bad", "OptimizationConfig", OptimizationConfigMetadata("x", "1"))
        with self.assertRaises(ValueError):
            CheckResult("", CheckStatus.FAIL)

    def test_registry_rejects_invalid_and_duplicates(self):
        registry = Registry()
        spec = registry.register_spec(
            ReplacementSpec(
                "p",
                Mechanism.REPLACE,
                "m.f",
                "m.replacement",
            )
        )
        group = registry.register_group(OptimizationGroup("g", ("p",)))
        with self.assertRaises(ValueError):
            registry.register_spec(spec)
        with self.assertRaises(ValueError):
            registry.register_group(group)
        self.assertIn("p", registry.specs)
        self.assertIn("g", registry.groups)
        with self.assertRaises(ValueError):
            ReplacementSpec("bad", Mechanism.REPLACE, "m.f", "")

    def test_context_handles_missing_git_dependency_and_bad_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(_run_git(["not-a-real-command"], Path(directory)))
            snapshot = detect_context(
                dependency_names=("definitely-not-installed-turbo_physai-test",),
                project_dir=Path(directory),
            )
        self.assertIsNone(
            snapshot.dependencies["definitely-not-installed-turbo_physai-test"]
        )
        with patch.dict(os.environ, {"RANK": "invalid"}):
            self.assertEqual(_int_env("RANK", 7), 7)

    def test_checker_helpers_cover_type_signature_and_values(self):
        self.assertEqual(
            checker_module._target_type_check(target, Mechanism.REPLACE),
            (True, "callable, class or property"),
        )
        self.assertEqual(
            checker_module._target_type_check(type, Mechanism.REPLACE),
            (True, "callable, class or property"),
        )
        self.assertEqual(
            checker_module._target_type_check(target, Mechanism.WRAPPER),
            (True, "callable"),
        )
        self.assertTrue(checker_module._signature_compatible(target, permissive))
        self.assertFalse(checker_module._signature_compatible(target, incompatible))
        self.assertFalse(
            checker_module._signature_compatible(target, adds_required_parameter)
        )
        self.assertFalse(
            checker_module._signature_compatible(target, keyword_only_extra)
        )
        self.assertTrue(
            checker_module._signature_compatible(all_argument_kinds, permissive)
        )
        self.assertIsNone(checker_module._signature_compatible(len, object()))

        class ReadOnly:
            @property
            def value(self):
                return 1

        class Compatible:
            @property
            def value(self):
                return 2

        class AddsSetter:
            @property
            def value(self):
                return 2

            @value.setter
            def value(self, new_value):
                del new_value

        self.assertEqual(
            checker_module._target_type_check(ReadOnly.value, Mechanism.REPLACE),
            (True, "callable, class or property"),
        )
        self.assertTrue(
            checker_module._signature_compatible(
                ReadOnly.value,
                Compatible.value,
            )
        )
        self.assertFalse(
            checker_module._signature_compatible(
                ReadOnly.value,
                AddsSetter.value,
            )
        )
        self.assertTrue(checker_module._version_matches("1.2.3", ("1.2.3",)))
        self.assertTrue(checker_module._version_matches("1.2.3", (">=1,<2",)))
        self.assertFalse(checker_module._version_matches(None, (">=1",)))
        self.assertFalse(checker_module._version_matches("invalid", (">=1",)))
        self.assertEqual(
            checker_module._trusted_values(
                {"ast_hashes": {"m.f": "ast-v1:x"}}, "ast_hashes", "m.f"
            ),
            ("ast-v1:x",),
        )
        self.assertEqual(
            checker_module._trusted_values(
                {"ast_hashes": {"m.f": 1}}, "ast_hashes", "m.f"
            ),
            (),
        )

    def test_checker_structural_failures(self):
        module = types.ModuleType("negative_checker")
        module.target = target
        module.alias = incompatible
        module.wrapper = permissive
        sys.modules[module.__name__] = module
        try:
            registry = Registry()
            registry.register_spec(
                ReplacementSpec(
                    "alias.replacement",
                    Mechanism.WRAPPER,
                    "negative_checker.target",
                    "negative_checker.wrapper",
                    aliases=("negative_checker.alias",),
                )
            )
            registry.register_spec(
                ReplacementSpec(
                    "missing.replacement",
                    Mechanism.WRAPPER,
                    "negative_checker.target",
                    "negative_checker.not_present",
                )
            )
            group = OptimizationGroup(
                "bad.group", ("unknown.replacement", "missing.replacement", "alias.replacement")
            )
            checks = Checker(registry, default_handlers()).check_group(
                group,
                OptimizationGroupConfig("bad.group"),
                EnvironmentSnapshot("3", "test", "python", "/tmp"),
            )
            codes = {item.code for item in checks}
            self.assertIn("registry.spec_missing", codes)
            self.assertIn("replacement.unresolved", codes)
            self.assertIn("alias.identity", codes)
        finally:
            sys.modules.pop(module.__name__, None)

    def test_environment_check_matrix(self):
        registry = Registry()
        checker = Checker(registry, default_handlers())
        environment = EnvironmentSnapshot(
            "3",
            "test",
            "python",
            "/tmp",
            dependencies=FrozenDict({"dep": "1"}),
            repository="repo",
            commit="abc",
            dirty=True,
            backend="hcu",
        )
        checks = tuple(
            checker.check_environment(
                environment,
                {
                    "dependencies": {"dep": [">=0.9,<2"]},
                    "commits": "abc",
                    "allow_dirty": True,
                    "backend": ["hcu"],
                    "repository": "repo",
                },
            )
        )
        self.assertTrue(all(item.status == CheckStatus.PASS for item in checks))
        advisory = tuple(
            checker.check_environment(
                environment,
                {"commits": ["different"]},
            )
        )
        self.assertEqual(len(advisory), 1)
        self.assertEqual(advisory[0].code, "project.commit")
        self.assertEqual(advisory[0].status, CheckStatus.WARNING)
        failed = tuple(
            checker.check_environment(
                environment,
                {
                    "dependencies": {"dep": ">=2"},
                    "allow_dirty": False,
                    "backend": "cuda",
                    "repository": ["other"],
                },
            )
        )
        self.assertTrue(all(item.status == CheckStatus.FAIL for item in failed))
        unknown = tuple(
            checker.check_environment(
                EnvironmentSnapshot("3", "test", "python", "/tmp"),
                {"allow_dirty": False},
            )
        )
        self.assertTrue(all(item.status == CheckStatus.UNKNOWN for item in unknown))
        self.assertTrue(all(item.overrideable for item in unknown))

    def test_handler_error_and_noop_paths(self):
        with self.assertRaises(HandlerError):
            resolve_attribute("invalid")
        with self.assertRaises(HandlerError):
            resolve_attribute("not_loaded.module.target", import_missing=False)

        module = types.ModuleType("negative_handler")
        module.target = target
        module.alias = incompatible
        module.mapping = {"value": 1}
        module.empty_mapping = {}
        module.object = types.SimpleNamespace()
        module.class_value = type
        module.function_value = target
        module.bad_wrapper = 1
        sys.modules[module.__name__] = module
        try:
            resolved = resolve_attribute("negative_handler.mapping.value")
            set_attribute(resolved, 2)
            self.assertEqual(module.mapping["value"], 2)
            with self.assertRaises(HandlerError):
                resolve_attribute("negative_handler.empty_mapping.missing.value")
            with self.assertRaises(HandlerError):
                resolve_attribute("negative_handler.object.missing.value")
            with self.assertRaises(HandlerError):
                resolve_attribute("negative_handler.object.missing")
            spec = ReplacementSpec(
                "p",
                Mechanism.REPLACE,
                "negative_handler.target",
                "negative_handler.missing",
            )
            with self.assertRaises(NotImplementedError):
                MechanismHandler().prepare(spec, {})
            noop_spec = ReplacementSpec(
                "noop",
                Mechanism.REPLACE,
                "negative_handler.target",
                "negative_handler.target",
            )
            alias_spec = ReplacementSpec(
                "alias",
                Mechanism.REPLACE,
                "negative_handler.target",
                "negative_handler.target",
                aliases=("negative_handler.alias",),
            )
            with self.assertRaises(HandlerError):
                default_handlers()[Mechanism.REPLACE].prepare(
                    alias_spec, {}
                )
            prepared = default_handlers()[Mechanism.REPLACE].prepare(
                noop_spec, {}
            )
            self.assertEqual(
                default_handlers()[Mechanism.REPLACE].apply(prepared),
                ("negative_handler.target",),
            )
            bad_function = ReplacementSpec(
                "bad.function",
                Mechanism.REPLACE,
                "negative_handler.target",
                "negative_handler.class_value",
            )
            with self.assertRaises(HandlerError):
                default_handlers()[Mechanism.REPLACE].prepare(
                    bad_function, {}
                )
            bad_class = ReplacementSpec(
                "bad.class",
                Mechanism.REPLACE,
                "negative_handler.class_value",
                "negative_handler.function_value",
            )
            with self.assertRaises(HandlerError):
                default_handlers()[Mechanism.REPLACE].prepare(
                    bad_class, {}
                )
            bad_wrapper = ReplacementSpec(
                "negative_handler.bad_wrapper",
                Mechanism.WRAPPER,
                "negative_handler.target",
                "negative_handler.bad_wrapper",
            )
            with self.assertRaises(HandlerError):
                default_handlers()[Mechanism.WRAPPER].prepare(
                    bad_wrapper, {}
                )
        finally:
            sys.modules.pop(module.__name__, None)

    def test_import_handler_invalid_nested_and_existing_paths(self):
        handler = ImportReplaceHandler()
        replacement_values = types.ModuleType("negative_import_replacements")
        replacement_values.bad = 1
        replacement = types.ModuleType("replacement")
        replacement_values.replacement = replacement
        sys.modules[replacement_values.__name__] = replacement_values
        missing_spec = ReplacementSpec(
            "import.negative",
            Mechanism.IMPORT_REPLACE,
            "negative_import.sub.op",
            "negative_import_replacements.missing",
        )
        with self.assertRaises(HandlerError):
            handler.prepare(missing_spec, {})
        invalid_spec = ReplacementSpec(
            "import.invalid",
            Mechanism.IMPORT_REPLACE,
            "negative_import.sub.op",
            "negative_import_replacements.bad",
        )
        with self.assertRaises(HandlerError):
            handler.prepare(invalid_spec, {})

        spec = ReplacementSpec(
            "import.valid",
            Mechanism.IMPORT_REPLACE,
            "negative_import.sub.op",
            "negative_import_replacements.replacement",
        )
        prepared = handler.prepare(spec, {})
        snapshot = handler.snapshot(prepared)
        handler.apply(prepared)
        self.assertIn("negative_import.sub.op", sys.modules)
        handler.restore(snapshot)
        self.assertNotIn("negative_import", sys.modules)

        parent = types.ModuleType("negative_import")
        original = types.ModuleType("negative_import.sub")
        parent.sub = original
        sys.modules["negative_import"] = parent
        sys.modules["negative_import.sub"] = original
        try:
            existing_spec = ReplacementSpec(
                "import.existing",
                Mechanism.IMPORT_REPLACE,
                "negative_import.sub",
                "negative_import_replacements.replacement",
            )
            prepared = handler.prepare(existing_spec, {})
            snapshot = handler.snapshot(prepared)
            handler.apply(prepared)
            self.assertIs(parent.sub, replacement)
            handler.restore(snapshot)
            self.assertIs(parent.sub, original)
            self.assertIs(sys.modules["negative_import.sub"], original)

            replacement_values.original = original
            same_object_spec = ReplacementSpec(
                "import.same_object",
                Mechanism.IMPORT_REPLACE,
                "negative_import.sub",
                "negative_import_replacements.original",
            )
            same_object = handler.prepare(same_object_spec, {})
            self.assertEqual(
                handler.apply(same_object), ("negative_import.sub",)
            )
        finally:
            sys.modules.pop("negative_import.sub", None)
            sys.modules.pop("negative_import", None)
            sys.modules.pop(replacement_values.__name__, None)

    def test_optimization_config_loader_and_schema_failures(self):
        with self.assertRaises(OptimizationConfigNotFoundError):
            resolve_optimization_config_path("/definitely/missing/config.yaml")
        invalid_values = [
            [],
            {},
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "Wrong",
                "metadata": {"id": "x", "version": "1"},
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "extends": "not-a-list",
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": [{"id": "g", "enabled": "yes"}],
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": [{"id": "g", "trust": {"ast_hashes": []}}],
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "model": {"name": 1},
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "compatibility": {"allow_dirty": "no"},
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "compatibility": {"dependencies": []},
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": "bad",
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": [{"id": ""}],
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": [{"id": "g"}, {"id": "g"}],
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": [{"id": "g", "requirement": "sometimes"}],
            },
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "x", "version": "1"},
                "optimization_groups": [{"id": "g", "options": {"callable": target}}],
            },
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(OptimizationConfigError):
                optimization_config_from_dict(value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "empty.yaml"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(OptimizationConfigError):
                load_optimization_config(empty)
            malformed = root / "malformed.yaml"
            malformed.write_text("[", encoding="utf-8")
            with self.assertRaises(OptimizationConfigError):
                load_optimization_config(malformed)

    def test_catalog_unknown_parent_cycle_and_duplicate(self):
        parent = optimization_config_from_dict(
            {
                "schema_version": "turbophysai/optimization-config/v1",
                "kind": "OptimizationConfig",
                "metadata": {"id": "parent", "version": "1"},
                "extends": ["parent"],
            }
        )
        catalog = OptimizationConfigCatalog({"parent": parent})
        with self.assertRaises(OptimizationConfigError):
            catalog.register(parent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "child.yaml"
            path.write_text(
                "schema_version: turbophysai/optimization-config/v1\nkind: OptimizationConfig\n"
                "metadata: {id: child, version: '1'}\nextends: [missing]\n",
                encoding="utf-8",
            )
            with self.assertRaises(OptimizationConfigError):
                load_optimization_config(path, catalog=catalog)
            path.write_text(
                "schema_version: turbophysai/optimization-config/v1\nkind: OptimizationConfig\n"
                "metadata: {id: child, version: '1'}\nextends: [parent]\n",
                encoding="utf-8",
            )
            with self.assertRaises(OptimizationConfigError):
                load_optimization_config(path, catalog=catalog)

    def test_report_write_error_is_public(self):
        with self.assertRaises(ReportWriteError) as raised:
            turbo_physai.apply(report_dir="/dev/null")
        self.assertIsNotNone(raised.exception.report)


if __name__ == "__main__":
    unittest.main()
