# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import _bz2
import sys
import tempfile
import types
import unittest
from pathlib import Path

from turbo_physai.engine.checking.evidence import ast_hash, source_hash
from turbo_physai.engine.contracts import (
    Mechanism,
    ReplacementSpec,
    RestoreStatus,
)
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.engine.execution.replacements.base import HandlerError


def golden_hash_target(value):
    return value + 1


class HashAndHandlersTest(unittest.TestCase):
    def test_v1_hashes_match_golden_values(self):
        self.assertEqual(
            source_hash(golden_hash_target),
            "source-v1:135da93859dbb958406aab96c4d94739ba709fa5e39462fddd302ae28d2b2829",
        )
        self.assertEqual(
            ast_hash(golden_hash_target),
            "ast-v1:93330de0f7b55ea6084a8fdcc1fbb42861fd687e7b19657c0107c8004c51d080",
        )

    def test_ast_hash_ignores_format_and_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.py"
            second = root / "second.py"
            first.write_text(
                "def calculate(value):\n    return value + 1\n", encoding="utf-8"
            )
            second.write_text(
                "def calculate( value ):\n    # formatting only\n    return value+1\n",
                encoding="utf-8",
            )
            third = root / "third.py"
            third.write_text(
                "def calculate(value):\n    return value + 2\n", encoding="utf-8"
            )
            modules = []
            for name, path in (
                ("hash_first", first),
                ("hash_second", second),
                ("hash_third", third),
            ):
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules.append(module)
            self.assertEqual(
                ast_hash(modules[0].calculate), ast_hash(modules[1].calculate)
            )
            self.assertNotEqual(
                source_hash(modules[0].calculate), source_hash(modules[1].calculate)
            )
            self.assertNotEqual(
                ast_hash(modules[0].calculate), ast_hash(modules[2].calculate)
            )

    def test_native_callable_uses_extension_artifact_hash(self):
        digest = source_hash(_bz2.BZ2Compressor)
        self.assertIsNotNone(digest)
        self.assertTrue(digest.startswith("artifact-v1:"), digest)
        self.assertIsNone(ast_hash(_bz2.BZ2Compressor))

    def test_replace_handler_infers_class(self):
        module = types.ModuleType("class_fake")

        class Original:
            pass

        class Replacement:
            pass

        module.Original = Original
        module.Replacement = Replacement
        sys.modules[module.__name__] = module
        try:
            spec = ReplacementSpec(
                "class.replacement",
                Mechanism.REPLACE,
                "class_fake.Original",
                "class_fake.Replacement",
            )
            handler = default_handlers()[Mechanism.REPLACE]
            prepared = handler.prepare(spec, {})
            self.assertEqual(prepared.spec.mechanism, Mechanism.REPLACE)
            snapshot = handler.snapshot(prepared)
            handler.apply(prepared)
            self.assertIs(module.Original, Replacement)
            handler.restore(snapshot)
            self.assertIs(module.Original, Original)
        finally:
            sys.modules.pop(module.__name__, None)

    def test_replace_handler_applies_and_restores_read_only_property(self):
        module = types.ModuleType("property_fake")

        class Target:
            @property
            def value(self):
                return 1

        def optimized_value(self):
            return 2

        original = Target.value
        module.Target = Target
        module.optimized_value = property(optimized_value)
        sys.modules[module.__name__] = module
        try:
            spec = ReplacementSpec(
                "property.replacement",
                Mechanism.REPLACE,
                "property_fake.Target.value",
                "property_fake.optimized_value",
            )
            handler = default_handlers()[Mechanism.REPLACE]
            prepared = handler.prepare(spec, {})
            snapshot = handler.snapshot(prepared)

            handler.apply(prepared)
            self.assertEqual(Target().value, 2)

            results = handler.restore(snapshot)
            self.assertTrue(
                all(item.status == RestoreStatus.RESTORED for item in results)
            )
            self.assertIs(Target.value, original)
            self.assertEqual(Target().value, 1)
        finally:
            sys.modules.pop(module.__name__, None)

    def test_replace_handler_rejects_function_for_property_target(self):
        module = types.ModuleType("property_type_fake")

        class Target:
            @property
            def value(self):
                return 1

        def optimized_value(self):
            return 2

        module.Target = Target
        module.optimized_value = optimized_value
        sys.modules[module.__name__] = module
        try:
            spec = ReplacementSpec(
                "property.invalid",
                Mechanism.REPLACE,
                "property_type_fake.Target.value",
                "property_type_fake.optimized_value",
            )
            with self.assertRaisesRegex(HandlerError, "must be a property"):
                default_handlers()[Mechanism.REPLACE].prepare(spec, {})
        finally:
            sys.modules.pop(module.__name__, None)

    def test_property_hash_covers_all_declared_accessors(self):
        class ReadOnly:
            @property
            def value(self):
                return 1

        class ReadWrite:
            @property
            def value(self):
                return 1

            @value.setter
            def value(self, new_value):
                del new_value

        self.assertIsNotNone(source_hash(ReadOnly.value))
        self.assertIsNotNone(ast_hash(ReadOnly.value))
        self.assertNotEqual(
            source_hash(ReadOnly.value), source_hash(ReadWrite.value)
        )
        self.assertNotEqual(ast_hash(ReadOnly.value), ast_hash(ReadWrite.value))

    def test_import_replace_handler_restores_sys_modules(self):
        for name in ("generated_pkg.op", "generated_pkg"):
            sys.modules.pop(name, None)
        replacement_module = types.ModuleType("import_replacement_fake")
        replacement_module.VALUE = 7
        sys.modules[replacement_module.__name__] = replacement_module
        spec = ReplacementSpec(
            "import.replacement",
            Mechanism.IMPORT_REPLACE,
            "generated_pkg.op",
            "import_replacement_fake",
        )
        handler = default_handlers()[Mechanism.IMPORT_REPLACE]
        try:
            prepared = handler.prepare(spec, {})
            self.assertEqual(replacement_module.__name__, "import_replacement_fake")
            snapshot = handler.snapshot(prepared)
            handler.apply(prepared)
            self.assertEqual(sys.modules["generated_pkg.op"].VALUE, 7)
            handler.restore(snapshot)
            self.assertNotIn("generated_pkg.op", sys.modules)
            self.assertNotIn("generated_pkg", sys.modules)
        finally:
            sys.modules.pop(replacement_module.__name__, None)

    def test_import_restore_detects_silent_parent_link_failure(self):
        original = types.ModuleType("sticky_pkg.op")
        replacement_module = types.ModuleType("sticky_replacement_fake")
        replacement_module.VALUE = 7

        class StickyParent(types.ModuleType):
            def __setattr__(self, name, value):
                if name == "op" and value is original:
                    return
                super().__setattr__(name, value)

        parent = StickyParent("sticky_pkg")
        types.ModuleType.__setattr__(parent, "op", original)
        sys.modules[parent.__name__] = parent
        sys.modules[original.__name__] = original
        sys.modules[replacement_module.__name__] = replacement_module
        spec = ReplacementSpec(
            "sticky.import.replacement",
            Mechanism.IMPORT_REPLACE,
            "sticky_pkg.op",
            "sticky_replacement_fake",
        )
        handler = default_handlers()[Mechanism.IMPORT_REPLACE]
        try:
            prepared = handler.prepare(spec, {})
            snapshot = handler.snapshot(prepared)
            handler.apply(prepared)
            replacement = parent.op

            results = handler.restore(snapshot)

            self.assertTrue(
                any(result.status == RestoreStatus.FAILED for result in results)
            )
            self.assertIs(sys.modules["sticky_pkg.op"], original)
            self.assertIs(parent.op, replacement)
        finally:
            for name in (
                replacement_module.__name__,
                original.__name__,
                parent.__name__,
            ):
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
