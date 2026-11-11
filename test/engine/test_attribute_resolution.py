# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import sys
import types
import unittest
from unittest.mock import patch

from turbo_physai.engine.contracts import Mechanism, ReplacementSpec, RestoreStatus
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.engine.execution.replacements.base import (
    MechanismHandler,
    resolve_attribute,
    set_attribute,
)


class AttributeResolutionTest(unittest.TestCase):
    def test_dynamic_class_attribute_can_be_replaced_and_restored(self):
        def original(value):
            return value + 1

        def replacement(value):
            return value + 2

        for import_missing in (False, True):
            with self.subTest(import_missing=import_missing):
                class DynamicMeta(type):
                    def __getattr__(cls, name):
                        if name == "run":
                            return original
                        raise AttributeError(name)

                class Target(metaclass=DynamicMeta):
                    pass

                module = types.ModuleType("dynamic_attribute_fake")
                module.Target = Target
                module.replacement = replacement
                spec = ReplacementSpec(
                    "dynamic.attribute", Mechanism.REPLACE,
                    "dynamic_attribute_fake.Target.run",
                    "dynamic_attribute_fake.replacement",
                )
                with patch.dict(sys.modules, {module.__name__: module}):
                    handler = default_handlers()[Mechanism.REPLACE]
                    self.assertNotIn("run", vars(Target))
                    prepared = handler.prepare(spec, {}, import_missing=import_missing)
                    snapshot = handler.snapshot(prepared)
                    self.assertIs(snapshot[0].original, original)
                    self.assertIsNone(snapshot[0].class_owned)
                    handler.apply(prepared)
                    self.assertEqual(Target.run(3), 5)
                    results = handler.restore(snapshot)
                    self.assertEqual(results[0].status, RestoreStatus.RESTORED)
                    self.assertIs(Target.run, original)
                    self.assertEqual(Target.run(3), 4)

    def test_static_descriptors_restore_binding_and_inheritance(self):
        def method(*args):
            return args

        def replacement(*args):
            return "optimized"

        descriptors = {
            "method": method,
            "classmethod": classmethod(method),
            "staticmethod": staticmethod(method),
        }
        for kind, descriptor in descriptors.items():
            for inherited in (False, True):
                with self.subTest(kind=kind, inherited=inherited):
                    base = type("Base", (), {"run": descriptor})
                    target = type("Target", (base,), {}) if inherited else base
                    instance = target()
                    expected = instance.run()
                    original_attributes = dict(vars(target))
                    module = types.ModuleType("descriptor_attribute_fake")
                    module.Target = target
                    with patch.dict(sys.modules, {module.__name__: module}):
                        resolved = resolve_attribute(
                            "descriptor_attribute_fake.Target.run",
                            import_missing=False,
                        )
                        set_attribute(resolved, replacement)
                        self.assertEqual(instance.run(), "optimized")
                        results = MechanismHandler().restore((resolved,))
                        self.assertEqual(results[0].status, RestoreStatus.RESTORED)
                        self.assertIs(inspect.getattr_static(target, "run"), descriptor)
                        self.assertEqual(dict(vars(target)), original_attributes)
                        self.assertEqual(instance.run(), expected)


if __name__ == "__main__":
    unittest.main()
