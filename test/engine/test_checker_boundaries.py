# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Malformed compatibility hooks must block a group without executing it."""

import importlib.metadata
import inspect
import os
import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

import turbo_physai.engine as engine
from turbo_physai.engine.checking import checker as checks
from turbo_physai.engine.contracts import (
    CheckStatus,
    CompatibilityContext,
    CompatibilityResult,
    EnvironmentSnapshot,
    Mechanism,
    OptimizationGroup,
    OptimizationGroupConfig,
    ReplacementSpec,
)
from turbo_physai.engine.definitions import declarations
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.errors import OptimizationConfigError
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.engine.execution.replacements.base import HandlerError
from test.engine.test_config_boundaries import config


def original(value, optional=None):
    return value


def replacement(value, optional=None):
    return value + 1


class CheckerBoundariesTest(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("checker_fixture")
        self.module.original = original
        self.module.replacement = replacement
        self.module.bad_condition = 1
        self.registry = Registry()
        self.spec = self.registry.register_spec(
            ReplacementSpec(
                "op",
                Mechanism.REPLACE,
                "checker_fixture.original",
                "checker_fixture.replacement",
            )
        )
        self.definition = self.registry.register_group(OptimizationGroup("g", ("op",)))
        self.entry = OptimizationGroupConfig("g")
        self.environment = EnvironmentSnapshot("3.10", "linux", "python", ".")
        self.checker = checks.Checker(self.registry, default_handlers())
        modules = patch.dict(sys.modules, {"checker_fixture": self.module})
        modules.start()
        self.addCleanup(modules.stop)

    def run_check(self, definition=None):
        return {
            r.code: r
            for r in self.checker.check_group(
                definition or self.definition, self.entry, self.environment
            )
        }

    def test_missing_handler_and_unexpected_preparation_failure_block_group(self):
        checker = checks.Checker(self.registry, {})
        result = checker.check_group(self.definition, self.entry, self.environment)
        self.assertEqual(result[0].code, "registry.handler_missing")
        self.assertEqual(result[0].status, CheckStatus.FAIL)
        handler = self.checker.handlers[Mechanism.REPLACE]
        for error, code in [
            (RuntimeError("backend load failed"), "replacement.load_error"),
            (HandlerError("replacement cannot be resolved"), "replacement.unresolved"),
        ]:
            with self.subTest(error=error), patch.object(
                handler, "prepare", side_effect=error
            ):
                results = self.run_check()
                self.assertEqual(results[code].status, CheckStatus.FAIL)
                self.assertIn(str(error), results[code].detail)
        self.assertIs(self.module.original, original)

    def test_invalid_custom_checks_are_reported_without_mutation(self):
        def raises(context):
            raise RuntimeError("hook exploded")

        cases = [
            ("missing", None, "compatibility.unresolved"),
            ("number", 1, "compatibility.invalid_check"),
            ("class_check", object, "compatibility.invalid_check"),
            ("throws", raises, "compatibility.error"),
            ("boolean", lambda context: True, "compatibility.invalid_result"),
        ]
        for name, hook, code in cases:
            if hook is not None:
                setattr(self.module, name, hook)
            with self.subTest(name=name):
                definition = replace(
                    self.definition, compatibility_check="checker_fixture." + name
                )
                result = self.run_check(definition)[code]
                self.assertEqual(result.status, CheckStatus.FAIL)
                self.assertIs(self.module.original, original)

    def test_source_identity_without_inspectable_source_is_unknown(self):
        (result,) = self.checker._source_identity_check(
            self.spec, {"source_hash": None, "ast_hash": None}, {}
        )
        self.assertEqual(result.status, CheckStatus.UNKNOWN)
        self.assertFalse(result.overrideable)
        self.assertFalse(checks._signature_compatible(property(original), original))
        self.assertFalse(
            checks._signature_compatible(property(original), property(lambda: None))
        )
        self.assertEqual(
            checks._target_type_check(object(), Mechanism.IMPORT_REPLACE),
            (True, "module path"),
        )
        self.assertEqual(
            checks._trusted_values(
                {"source_hashes": {"target": None}}, "source_hashes", "target"
            ),
            (),
        )
        self.assertEqual(checks._values(None), ())
        self.assertFalse(checks._version_matches("bad version", [">=2", "~invalid"]))

        def variadic(__turbo_physai_extra__=None, *, optional=None, **kwargs):
            pass

        for args, kwargs in checks._representative_calls(inspect.signature(variadic)):
            inspect.signature(variadic).bind(*args, **kwargs)
        self.assertTrue(checks._signature_compatible(variadic, variadic))

    def test_invalid_run_id_and_group_overrides_fail_before_application(self):
        with patch.dict(os.environ, {"TURBO_PHYSAI_RUN_ID": "a" * 32}):
            self.assertEqual(engine._resolve_run_id(), "a" * 32)
        for value in ("", "z" * 32, "A" * 32, "a" * 31):
            with self.subTest(value=value), patch.dict(
                os.environ, {"TURBO_PHYSAI_RUN_ID": value}
            ):
                with self.assertRaises(OptimizationConfigError):
                    engine._resolve_run_id()
        selected = config([{"id": "g"}, {"id": "dependent"}])
        for values in ("g", [""], [None]):
            with self.subTest(values=values), self.assertRaises(
                OptimizationConfigError
            ):
                engine._validate_group_ids(selected, values, name="force_groups")
        self.registry.register_group(
            OptimizationGroup("dependent", ("op",), depends_on=("g",))
        )
        with self.assertRaisesRegex(OptimizationConfigError, "depend on disabled"):
            engine._apply_group_overrides(
                selected, self.registry, ("dependent",), ("g",)
            )

    def test_declarations_validate_before_registering_and_allow_idempotence(self):
        registry = Registry()
        declared = declarations.group(
            "new", declarations.replace("a.fn", "b.fn"), registry=registry
        )
        declared.register(registry)
        self.assertEqual(len(registry.specs), 1)
        spec = declared.specs[0]
        conflicting = replace(declared, specs=(replace(spec, replacement="c.fn"),))
        with self.assertRaisesRegex(ValueError, "duplicate ReplacementSpec"):
            conflicting.register(registry)
        self.assertEqual(registry.get_spec(spec.replacement_id), spec)
        invalid_calls = [
            lambda: declarations.replace("", "b.fn"),
            lambda: declarations.replace("a.fn", "b.fn", runtime_condition=""),
            lambda: declarations.import_alias("a", "b.c", "alias"),
            lambda: declarations.optional_import("a"),
            lambda: declarations.registry_override(
                "module", "registry", names=["Op", "Op"]
            ),
            lambda: declarations.group("empty", registry=registry),
            lambda: declarations.group("invalid", "raw_spec", registry=registry),
            lambda: OptimizationGroup("g", ("op",), depends_on=("a", "a")),
            lambda: OptimizationGroup("g", ("op",), compatibility_check=""),
            lambda: CompatibilityResult("yes"),
        ]
        for invalid in invalid_calls:
            with self.subTest(invalid=invalid), self.assertRaises(
                (ValueError, TypeError)
            ):
                invalid()
        with patch.object(importlib.metadata, "version", return_value="1.2.3"):
            self.assertEqual(CompatibilityContext.package_version("fixture"), "1.2.3")
        with patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            self.assertIsNone(CompatibilityContext.package_version("fixture"))
