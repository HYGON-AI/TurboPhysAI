# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
import types
import unittest
from unittest import mock

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.definitions import group, replace, replace_import, wrap
from turbo_physai.engine.execution.replacements import HandlerError, default_handlers


def original_function(value):
    return value


def replacement_function(value):
    return value + 1


def wrapper(original, options):
    def wrapped(value):
        return original(value) + options.get("offset", 0)

    return wrapped


def positive_value(value):
    return value > 0


def condition_failure(value):
    raise RuntimeError(f"condition failed for {value}")


def invalid_condition_result(value):
    return value


class OriginalClass:
    pass


class ReplacementClass:
    pass


class SimplifiedDeclarationTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        self.declaration = group(
            "customer.simple",
            replace("fake_original.function", "fake_replacement.function"),
            replace("fake_original.Class", "fake_replacement.Class"),
            wrap("fake_original.wrapped", "fake_replacement.wrapper"),
            registry=self.registry,
        )

    def test_group_generates_stable_internal_contracts(self):
        registered = self.registry.get_group("customer.simple")
        self.assertEqual(registered, self.declaration.definition)
        self.assertEqual(len(registered.members), 3)
        self.assertEqual(
            tuple(
                self.registry.get_spec(replacement_id).mechanism
                for replacement_id in registered.members
            ),
            (Mechanism.REPLACE, Mechanism.REPLACE, Mechanism.WRAPPER),
        )
        self.assertEqual(
            registered.members,
            group(
                "customer.simple",
                replace("fake_original.function", "fake_replacement.function"),
                replace("fake_original.Class", "fake_replacement.Class"),
                wrap("fake_original.wrapped", "fake_replacement.wrapper"),
                registry=Registry(),
            ).definition.members,
        )

    def test_group_declares_only_direct_dependencies(self):
        declaration = group(
            "customer.dependent",
            replace("fake_original.extra", "fake_replacement.extra"),
            depends_on=("customer.simple",),
            compatibility_check="customer.checks.compatible",
            registry=self.registry,
        )
        self.assertEqual(
            declaration.definition.depends_on,
            ("customer.simple",),
        )
        self.assertEqual(
            declaration.definition.compatibility_check,
            "customer.checks.compatible",
        )

    def test_replace_infers_function_and_class_after_resolution(self):
        original = types.ModuleType("fake_original")
        original.function = original_function
        original.Class = OriginalClass
        original.wrapped = original_function
        replacement = types.ModuleType("fake_replacement")
        replacement.function = replacement_function
        replacement.Class = ReplacementClass
        replacement.wrapper = wrapper
        handlers = default_handlers()
        specs = self.declaration.specs
        with mock.patch.dict(
            sys.modules,
            {"fake_original": original, "fake_replacement": replacement},
        ):
            function = handlers[Mechanism.REPLACE].prepare(specs[0], {})
            klass = handlers[Mechanism.REPLACE].prepare(specs[1], {})
            wrapped = handlers[Mechanism.WRAPPER].prepare(
                specs[2], {"offset": 2}
            )
        self.assertEqual(function.spec.mechanism, Mechanism.REPLACE)
        self.assertEqual(klass.spec.mechanism, Mechanism.REPLACE)
        self.assertEqual(wrapped.replacement(3), 5)

    def test_replace_accepts_native_callable_target(self):
        original = types.ModuleType("fake_native_original")
        original.operator = math.sin
        replacement = types.ModuleType("fake_native_replacement")
        replacement.operator = replacement_function
        declaration = group(
            "backend.native",
            replace(
                "fake_native_original.operator",
                "fake_native_replacement.operator",
            ),
            registry=Registry(),
        )
        with mock.patch.dict(
            sys.modules,
            {
                original.__name__: original,
                replacement.__name__: replacement,
            },
        ):
            prepared = default_handlers()[Mechanism.REPLACE].prepare(
                declaration.specs[0], {}
            )
        self.assertEqual(prepared.spec.mechanism, Mechanism.REPLACE)

    def test_replace_import_uses_module_handler_and_restores_import_state(self):
        replacement_module = types.ModuleType("catalog_import_replacement")
        replacement_module.VALUE = 7
        declaration = group(
            "model.private_extension",
            replace_import(
                "catalog_private_extension.operator",
                "catalog_import_replacement",
            ),
            registry=Registry(),
        )
        spec = declaration.specs[0]
        self.assertEqual(spec.mechanism, Mechanism.IMPORT_REPLACE)
        self.assertEqual(spec.aliases, ())
        self.assertIsNone(spec.runtime_condition)

        handler = default_handlers()[Mechanism.IMPORT_REPLACE]
        with mock.patch.dict(
            sys.modules,
            {replacement_module.__name__: replacement_module},
        ):
            prepared = handler.prepare(spec, {})
            snapshot = handler.snapshot(prepared)
            self.assertEqual(handler.apply(prepared), (spec.target,))
            self.assertIs(sys.modules[spec.target], replacement_module)
            self.assertIs(
                sys.modules["catalog_private_extension"].operator,
                replacement_module,
            )
            restore_results = handler.restore(snapshot)
            self.assertTrue(
                all(
                    result.status.value == "restored"
                    for result in restore_results
                )
            )
            self.assertNotIn(spec.target, sys.modules)
            self.assertNotIn("catalog_private_extension", sys.modules)

    def test_replace_import_is_exported_from_public_api(self):
        import turbo_physai

        self.assertIs(turbo_physai.replace_import, replace_import)

    def test_runtime_condition_dispatches_between_optimized_and_original(self):
        original = types.ModuleType("conditional_original")
        original.function = original_function
        original.alias = original_function
        replacement = types.ModuleType("conditional_replacement")
        replacement.function = replacement_function
        replacement.positive = positive_value
        declaration = group(
            "backend.conditional",
            replace(
                "conditional_original.function",
                "conditional_replacement.function",
                aliases=("conditional_original.alias",),
                runtime_condition="conditional_replacement.positive",
            ),
            registry=Registry(),
        )
        spec = declaration.specs[0]
        self.assertEqual(
            spec.runtime_condition,
            "conditional_replacement.positive",
        )
        with mock.patch.dict(
            sys.modules,
            {
                original.__name__: original,
                replacement.__name__: replacement,
            },
        ):
            handler = default_handlers()[Mechanism.REPLACE]
            prepared = handler.prepare(spec, {})
            self.assertEqual(prepared.replacement(-2), -2)
            self.assertEqual(prepared.replacement(2), 3)
            handler.apply(prepared)
            self.assertIs(original.function, original.alias)
            self.assertEqual(original.function(-3), -3)
            self.assertEqual(original.function(3), 4)
            handler.restore(handler.snapshot(prepared))
            self.assertIs(original.function, original_function)
            self.assertIs(original.alias, original_function)

    def test_runtime_condition_is_not_evaluated_during_prepare(self):
        original = types.ModuleType("condition_error_original")
        original.function = original_function
        replacement = types.ModuleType("condition_error_replacement")
        replacement.function = replacement_function
        replacement.condition = condition_failure
        declaration = group(
            "backend.condition_error",
            replace(
                "condition_error_original.function",
                "condition_error_replacement.function",
                runtime_condition="condition_error_replacement.condition",
            ),
            registry=Registry(),
        )
        with mock.patch.dict(
            sys.modules,
            {
                original.__name__: original,
                replacement.__name__: replacement,
            },
        ):
            prepared = default_handlers()[Mechanism.REPLACE].prepare(
                declaration.specs[0], {}
            )
            with self.assertRaisesRegex(RuntimeError, "condition failed"):
                prepared.replacement(1)

    def test_runtime_condition_requires_boolean_result(self):
        original = types.ModuleType("condition_result_original")
        original.function = original_function
        replacement = types.ModuleType("condition_result_replacement")
        replacement.function = replacement_function
        replacement.condition = invalid_condition_result
        declaration = group(
            "backend.condition_result",
            replace(
                "condition_result_original.function",
                "condition_result_replacement.function",
                runtime_condition="condition_result_replacement.condition",
            ),
            registry=Registry(),
        )
        with mock.patch.dict(
            sys.modules,
            {
                original.__name__: original,
                replacement.__name__: replacement,
            },
        ):
            prepared = default_handlers()[Mechanism.REPLACE].prepare(
                declaration.specs[0], {}
            )
            with self.assertRaisesRegex(
                TypeError,
                "runtime_condition must return bool",
            ):
                prepared.replacement(1)

    def test_runtime_condition_wraps_wrapper_result(self):
        original = types.ModuleType("conditional_wrapper_original")
        original.function = original_function
        replacement = types.ModuleType("conditional_wrapper_replacement")
        replacement.wrapper = wrapper
        replacement.positive = positive_value
        declaration = group(
            "model.conditional_wrapper",
            wrap(
                "conditional_wrapper_original.function",
                "conditional_wrapper_replacement.wrapper",
                runtime_condition="conditional_wrapper_replacement.positive",
            ),
            registry=Registry(),
        )
        with mock.patch.dict(
            sys.modules,
            {
                original.__name__: original,
                replacement.__name__: replacement,
            },
        ):
            prepared = default_handlers()[Mechanism.WRAPPER].prepare(
                declaration.specs[0], {"offset": 5}
            )
            self.assertEqual(prepared.replacement(-1), -1)
            self.assertEqual(prepared.replacement(1), 6)

    def test_runtime_condition_rejects_class_target(self):
        original = types.ModuleType("conditional_class_original")
        original.Class = OriginalClass
        replacement = types.ModuleType("conditional_class_replacement")
        replacement.Class = ReplacementClass
        replacement.positive = positive_value
        declaration = group(
            "backend.conditional_class",
            replace(
                "conditional_class_original.Class",
                "conditional_class_replacement.Class",
                runtime_condition="conditional_class_replacement.positive",
            ),
            registry=Registry(),
        )
        with mock.patch.dict(
            sys.modules,
            {
                original.__name__: original,
                replacement.__name__: replacement,
            },
        ):
            with self.assertRaisesRegex(
                HandlerError,
                "does not support class targets",
            ):
                default_handlers()[Mechanism.REPLACE].prepare(
                    declaration.specs[0], {}
                )

    def test_group_conflict_does_not_leave_orphan_specs(self):
        specs_before = self.registry.specs
        groups_before = self.registry.groups

        with self.assertRaisesRegex(ValueError, "duplicate OptimizationGroup id"):
            group(
                "customer.simple",
                replace("fake_original.function", "fake_replacement.function"),
                replace("fake_original.extra", "fake_replacement.extra"),
                registry=self.registry,
            )

        self.assertEqual(self.registry.specs, specs_before)
        self.assertEqual(self.registry.groups, groups_before)
        self.assertIsNone(self.registry.get_spec("customer.simple.extra"))

    def test_group_rejects_conflicting_or_duplicate_members_before_registering(self):
        for replacement, code in (
            ("fake_replacement.other", "intra_group_conflict"),
            ("fake_replacement.function", "intra_group_duplicate"),
        ):
            with self.subTest(code=code):
                registry = Registry()
                with self.assertRaisesRegex(ValueError, code):
                    group(
                        "customer.invalid",
                        replace(
                            "fake_original.function",
                            "fake_replacement.function",
                        ),
                        replace("fake_original.function", replacement),
                        registry=registry,
                    )
                self.assertEqual(registry.specs, {})
                self.assertEqual(registry.groups, {})

    def test_different_runtime_conditions_are_not_treated_as_duplicates(self):
        registry = Registry()
        with self.assertRaisesRegex(ValueError, "intra_group_conflict"):
            group(
                "customer.condition_conflict",
                replace(
                    "fake_original.function",
                    "fake_replacement.function",
                    runtime_condition="fake_replacement.positive",
                ),
                replace(
                    "fake_original.function",
                    "fake_replacement.function",
                    runtime_condition="fake_replacement.other_condition",
                ),
                registry=registry,
            )
        self.assertEqual(registry.specs, {})
        self.assertEqual(registry.groups, {})


if __name__ == "__main__":
    unittest.main()
