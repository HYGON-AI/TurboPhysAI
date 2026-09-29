# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Import compatibility is transactional even when modules or registries fail."""

import importlib
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from turbo_physai.engine.contracts import Mechanism, ReplacementSpec, RestoreStatus
from turbo_physai.engine.execution.replacements.base import HandlerError
from turbo_physai.engine.execution.replacements.import_alias import ImportAliasHandler
from turbo_physai.engine.execution.replacements.import_replace import (
    ImportReplaceHandler,
)
from turbo_physai.engine.execution.replacements.optional_import import (
    OptionalImportHandler,
)
from turbo_physai.engine.execution.replacements.registry_override import (
    RegistryOverrideHandler,
)


class Registry:
    def __init__(self):
        self._module_dict = {"Op": object()}
        self.calls = []

    def _register_module(
        self, module_class=None, module_name=None, force=False, **kwargs
    ):
        module_class = module_class or kwargs["module"]
        names = module_name or module_class.__name__
        names = [names] if isinstance(names, str) else names
        self.calls.append((tuple(names), force))
        for name in names:
            if name in self._module_dict and not force:
                raise KeyError(name)
            self._module_dict[name] = module_class


class RefusesMutation(types.ModuleType):
    def __setattr__(self, name, value):
        if name == "alias":
            raise RuntimeError("read-only alias")
        super().__setattr__(name, value)


class ImportTransactionsTest(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("transaction_fixture")
        self.module.source = object()
        self.modules = patch.dict(sys.modules, {self.module.__name__: self.module})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_alias_rejects_cross_module_missing_source_and_intervening_mutation(self):
        handler = ImportAliasHandler()
        spec = ReplacementSpec(
            "alias",
            Mechanism.IMPORT_ALIAS,
            "transaction_fixture.alias",
            "transaction_fixture.source",
        )
        for bad in (
            replace(spec, target="alias"),
            replace(spec, replacement="other.source"),
            replace(spec, replacement="transaction_fixture.absent"),
        ):
            with self.subTest(spec=bad), self.assertRaises(HandlerError):
                handler.prepare(bad, {}, import_missing=False)
        prepared = handler.prepare(spec, {})
        self.module.alias = object()
        with self.assertRaisesRegex(HandlerError, "changed after preparation"):
            handler.apply(prepared)
        del self.module.alias
        self.module.alias = self.module.source
        prepared = handler.prepare(spec, {}, import_missing=False)
        snapshot = handler.snapshot(prepared)
        handler.apply(prepared)
        self.module.alias = object()
        self.assertEqual(handler.restore(snapshot)[0].status, RestoreStatus.RESTORED)
        self.assertIs(self.module.alias, self.module.source)
        blocked = RefusesMutation("blocked")
        result = handler.restore(replace(snapshot, module=blocked))
        self.assertEqual(result[0].status, RestoreStatus.FAILED)
        self.assertIn("read-only", result[0].error)

    def test_optional_import_restores_existing_parent_attribute_and_absent_child(self):
        handler = OptionalImportHandler()
        spec = ReplacementSpec(
            "optional",
            Mechanism.OPTIONAL_IMPORT,
            "transaction_fixture.alias",
            "transaction_fixture.alias",
        )
        with self.assertRaises(HandlerError):
            handler.prepare(replace(spec, replacement="elsewhere"), {})
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.module.alias = self.module.source
                prepared = handler.prepare(spec, {})
                snapshot = handler.snapshot(prepared)
                handler.apply(prepared)
                placeholder = sys.modules[spec.target]
                self.assertIs(self.module.alias, placeholder)
                self.assertTrue(placeholder.__turbo_physai_optional_import__)
                handler.apply(prepared)
                self.assertIs(sys.modules[spec.target], placeholder)
                self.assertEqual(
                    handler.restore(snapshot)[0].status, RestoreStatus.RESTORED
                )
                self.assertNotIn(spec.target, sys.modules)
                if existing:
                    self.assertIs(self.module.alias, self.module.source)
                else:
                    self.assertFalse(hasattr(self.module, "alias"))
        sys.modules[spec.target] = self.module.source
        snapshot = handler.snapshot(prepared)
        handler.apply(prepared)
        handler.restore(snapshot)
        self.assertIs(sys.modules[spec.target], self.module.source)
        blocked = RefusesMutation("blocked")
        result = handler.restore(
            replace(snapshot, parent=blocked, parent_had_child=True)
        )
        self.assertEqual(result[0].status, RestoreStatus.FAILED)
        self.assertIn("read-only", result[0].error)

    def test_module_replace_restores_parent_links_and_reports_restore_failure(self):
        handler = ImportReplaceHandler()
        replacement = types.ModuleType("replacement")
        self.module.replacement = replacement
        spec = ReplacementSpec(
            "module",
            Mechanism.IMPORT_REPLACE,
            "transaction_fixture.alias",
            "transaction_fixture.replacement",
        )
        with self.assertRaisesRegex(HandlerError, "runtime_condition"):
            handler.prepare(replace(spec, runtime_condition="x.condition"), {})
        prepared = handler.prepare(spec, {})
        snapshot = handler.snapshot(prepared)
        handler.apply(prepared)
        self.assertIs(self.module.alias, replacement)
        handler.restore(snapshot)
        self.assertFalse(hasattr(self.module, "alias"))
        self.assertNotIn(spec.target, sys.modules)
        blocked = RefusesMutation("blocked")
        snapshot = replace(snapshot, parent_links=((blocked, "alias", True, object()),))
        result = handler.restore(snapshot)
        self.assertEqual(result[0].status, RestoreStatus.FAILED)
        self.assertEqual(result[-1].status, RestoreStatus.RESTORED)

    def test_registry_limits_forced_registration_and_restores_local_hook(self):
        variants = (
            "registry._register_module(Op)",
            "registry._register_module(Op, 'Op', False)",
            "registry._register_module(module=Op, module_name=['Op'])",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(sys, "path", [str(root)] + sys.path):
                for index, call in enumerate(variants):
                    with self.subTest(call=call):
                        name = f"transaction_override_{index}"
                        registry = Registry()
                        # Framework instances can override their registration hook locally.
                        registry._register_module = registry._register_module
                        original_hook = registry._register_module
                        original_entry = registry._module_dict["Op"]
                        self.module.registry = registry
                        (root / (name + ".py")).write_text(
                            "from transaction_fixture import registry\nclass Op: pass\n"
                            + call
                            + "\n"
                        )
                        importlib.invalidate_caches()
                        spec = ReplacementSpec(
                            "registry",
                            Mechanism.REGISTRY_OVERRIDE,
                            name,
                            "transaction_fixture.registry",
                            mechanism_options={"names": ["Op"]},
                        )
                        handler = RegistryOverrideHandler()
                        prepared = handler.prepare(spec, {})
                        snapshot = handler.snapshot(prepared)
                        handler.apply(prepared)
                        self.assertEqual(registry.calls, [(("Op",), True)])
                        self.assertEqual(registry._module_dict["Op"].__module__, name)
                        self.assertIs(registry._register_module, original_hook)
                        # Reusing an already imported, correctly overridden module is safe.
                        self.assertEqual(
                            handler.apply(prepared),
                            ("transaction_fixture.registry.Op",),
                        )
                        results = handler.restore(snapshot)
                        self.assertTrue(
                            all(r.status == RestoreStatus.RESTORED for r in results)
                        )
                        self.assertIs(registry._module_dict["Op"], original_entry)
                        self.assertIs(registry._register_module, original_hook)
                        self.assertNotIn(name, sys.modules)

    def test_registry_rejects_unexpected_or_incomplete_overrides(self):
        handler = RegistryOverrideHandler()
        registry = Registry()
        self.module.registry = registry
        spec = ReplacementSpec(
            "registry",
            Mechanism.REGISTRY_OVERRIDE,
            "transaction_bad_override",
            "transaction_fixture.registry",
            mechanism_options={"names": ["Op"]},
        )
        prepared = handler.prepare(spec, {})
        with patch.dict(sys.modules, {spec.target: types.ModuleType(spec.target)}):
            with self.assertRaisesRegex(HandlerError, "without expected overrides"):
                handler.apply(prepared)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sys, "path", [directory] + sys.path
        ):
            path = Path(directory) / (spec.target + ".py")
            path.write_text(
                "from transaction_fixture import registry\nclass Other: pass\nregistry._register_module(Other)\n"
            )
            importlib.invalidate_caches()
            snapshot = handler.snapshot(prepared)
            with self.assertRaisesRegex(HandlerError, "did not override all"):
                handler.apply(prepared)
            self.assertEqual(registry.calls, [(("Other",), False)])
            handler.restore(snapshot)
            self.assertNotIn("Other", registry._module_dict)
            self.assertNotIn(spec.target, sys.modules)
        for value, message in [
            (object(), "mutable"),
            (types.SimpleNamespace(module_dict={}), "_register_module"),
        ]:
            self.module.registry = value
            with self.subTest(value=value), self.assertRaisesRegex(
                HandlerError, message
            ):
                handler.prepare(spec, {})
        self.module.registry = registry
        for names in ([], [""], [1]):
            with self.subTest(names=names), self.assertRaisesRegex(
                HandlerError, "names are required"
            ):
                handler.prepare(replace(spec, mechanism_options={"names": names}), {})
        snapshot = handler.snapshot(prepared)
        registry._module_dict = None
        result = handler.restore(snapshot)
        self.assertEqual(result[0].status, RestoreStatus.FAILED)
        self.assertEqual(result[1].status, RestoreStatus.RESTORED)
