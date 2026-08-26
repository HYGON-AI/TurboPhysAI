# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine import apply, check
from turbo_physai.engine.definitions import (
    group,
    import_alias,
    optional_import,
    registry_override,
    replace_import,
)
from turbo_physai.engine.definitions.registry import Registry
from turbo_physai.engine.execution.replacements import default_handlers


class FakeRegistry:
    def __init__(self):
        self._module_dict = {}

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class, module_name=None, force=False):
        names = module_name or module_class.__name__
        if isinstance(names, str):
            names = (names,)
        for name in names:
            if not force and name in self._module_dict:
                raise KeyError(f"{name} is already registered")
            self._module_dict[name] = module_class

    def register_module(self, name=None, force=False):
        def decorate(module_class):
            self._register_module(
                module_class=module_class,
                module_name=name,
                force=force,
            )
            return module_class

        return decorate


class ImportCompatibilityTest(unittest.TestCase):
    def test_declarations_are_public_and_stable(self):
        from turbo_physai import compatibility

        registry = Registry()
        declaration = group(
            "model.import_compatibility",
            import_alias("vendor.mha", "MHA", "FlashMHA"),
            optional_import("model.optional_extension"),
            registry_override(
                "model.sparse",
                "framework.CONV_LAYERS",
                names=("SparseConv3d",),
            ),
            registry=registry,
        )
        self.assertIs(compatibility.import_alias, import_alias)
        self.assertIs(compatibility.optional_import, optional_import)
        self.assertIs(compatibility.registry_override, registry_override)
        self.assertEqual(
            tuple(spec.mechanism for spec in declaration.specs),
            (
                Mechanism.IMPORT_ALIAS,
                Mechanism.OPTIONAL_IMPORT,
                Mechanism.REGISTRY_OVERRIDE,
            ),
        )
        self.assertEqual(
            tuple(declaration.specs[2].mechanism_options["names"]),
            ("SparseConv3d",),
        )

    def test_import_alias_and_optional_import_restore(self):
        vendor = types.ModuleType("vendor.mha")

        class MHA:
            pass

        vendor.MHA = MHA
        parent = types.ModuleType("model")
        declarations = group(
            "model.imports",
            import_alias("vendor.mha", "MHA", "FlashMHA"),
            optional_import("model.optional_extension"),
            registry=Registry(),
        )
        handlers = default_handlers()
        with mock.patch.dict(
            sys.modules,
            {"vendor.mha": vendor, "model": parent},
            clear=False,
        ):
            prepared = [
                (handlers[spec.mechanism], handlers[spec.mechanism].prepare(spec, {}))
                for spec in declarations.specs
            ]
            snapshots = [handler.snapshot(item) for handler, item in prepared]
            for handler, item in prepared:
                handler.apply(item)
            self.assertIs(vendor.FlashMHA, MHA)
            placeholder = importlib.import_module("model.optional_extension")
            self.assertTrue(placeholder.__turbo_physai_optional_import__)
            for (handler, _), snapshot in reversed(tuple(zip(prepared, snapshots))):
                handler.restore(snapshot)
            self.assertFalse(hasattr(vendor, "FlashMHA"))
            self.assertNotIn("model.optional_extension", sys.modules)

    def test_import_alias_can_be_exposed_through_module_replacement(self):
        vendor = types.ModuleType("vendor.modules.mha")

        class MHA:
            pass

        vendor.MHA = MHA
        declarations = group(
            "model.cross_module_alias",
            import_alias("vendor.modules.mha", "MHA", "FlashMHA"),
            replace_import("vendor.flash_attention", "vendor.modules.mha"),
            registry=Registry(),
        )
        handlers = default_handlers()
        with mock.patch.dict(
            sys.modules,
            {"vendor.modules.mha": vendor},
            clear=False,
        ):
            prepared = [
                (handlers[spec.mechanism], handlers[spec.mechanism].prepare(spec, {}))
                for spec in declarations.specs
            ]
            snapshots = [handler.snapshot(item) for handler, item in prepared]
            for handler, item in prepared:
                handler.apply(item)
            exported = importlib.import_module("vendor.flash_attention")
            self.assertIs(exported.FlashMHA, MHA)
            for (handler, _), snapshot in reversed(tuple(zip(prepared, snapshots))):
                handler.restore(snapshot)
            self.assertNotIn("vendor.flash_attention", sys.modules)
            self.assertFalse(hasattr(vendor, "FlashMHA"))

    def test_registry_override_is_allowlisted_and_restorable(self):
        framework = types.ModuleType("framework")
        registry = FakeRegistry()

        class ExistingSparseConv3d:
            pass

        registry.module_dict["SparseConv3d"] = ExistingSparseConv3d
        framework.CONV_LAYERS = registry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "model"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "sparse.py").write_text(
                textwrap.dedent(
                    """
                    from framework import CONV_LAYERS

                    @CONV_LAYERS.register_module()
                    class SparseConv3d:
                        pass
                    """
                ),
                encoding="utf-8",
            )
            declaration = group(
                "model.registry",
                registry_override(
                    "model.sparse",
                    "framework.CONV_LAYERS",
                    names=("SparseConv3d",),
                ),
                registry=Registry(),
            )
            handler = default_handlers()[Mechanism.REGISTRY_OVERRIDE]
            with mock.patch.dict(sys.modules, {"framework": framework}, clear=False):
                sys.path.insert(0, str(root))
                try:
                    prepared = handler.prepare(declaration.specs[0], {})
                    snapshot = handler.snapshot(prepared)
                    changed = handler.apply(prepared)
                    self.assertEqual(
                        registry.module_dict["SparseConv3d"].__module__,
                        "model.sparse",
                    )
                    self.assertEqual(
                        changed,
                        ("framework.CONV_LAYERS.SparseConv3d",),
                    )
                    handler.restore(snapshot)
                    self.assertIs(
                        registry.module_dict["SparseConv3d"],
                        ExistingSparseConv3d,
                    )
                    self.assertNotIn("model.sparse", sys.modules)
                finally:
                    sys.path.remove(str(root))
                    sys.modules.pop("model", None)
                    sys.modules.pop("model.sparse", None)

    def test_registry_override_does_not_force_unlisted_names(self):
        framework = types.ModuleType("framework_reject")
        registry = FakeRegistry()
        registry.module_dict["Unexpected"] = object()
        framework.CONV_LAYERS = registry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "reject_model"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "sparse.py").write_text(
                "from framework_reject import CONV_LAYERS\n"
                "@CONV_LAYERS.register_module()\n"
                "class Unexpected: pass\n",
                encoding="utf-8",
            )
            declaration = group(
                "model.registry.reject",
                registry_override(
                    "reject_model.sparse",
                    "framework_reject.CONV_LAYERS",
                    names=("SparseConv3d",),
                ),
                registry=Registry(),
            )
            handler = default_handlers()[Mechanism.REGISTRY_OVERRIDE]
            with mock.patch.dict(
                sys.modules, {"framework_reject": framework}, clear=False
            ):
                sys.path.insert(0, str(root))
                try:
                    prepared = handler.prepare(declaration.specs[0], {})
                    snapshot = handler.snapshot(prepared)
                    with self.assertRaises(KeyError):
                        handler.apply(prepared)
                    handler.restore(snapshot)
                finally:
                    sys.path.remove(str(root))
                    sys.modules.pop("reject_model", None)
                    sys.modules.pop("reject_model.sparse", None)

    def test_engine_applies_import_compatibility_before_normal_targets(self):
        vendor = types.ModuleType("stage_vendor")

        class MHA:
            pass

        vendor.MHA = MHA
        framework = types.ModuleType("stage_framework")
        external_registry = FakeRegistry()

        class ExistingSparseConv3d:
            pass

        external_registry.module_dict["SparseConv3d"] = ExistingSparseConv3d
        framework.CONV_LAYERS = external_registry
        registry = Registry()
        group(
            "stage.import_compatibility",
            import_alias("stage_vendor", "MHA", "FlashMHA"),
            optional_import("stage_model.optional_extension"),
            registry_override(
                "stage_model.sparse",
                "stage_framework.CONV_LAYERS",
                names=("SparseConv3d",),
            ),
            registry=registry,
        )
        from turbo_physai.engine.definitions import replace

        group(
            "stage.runtime",
            replace("stage_model.target", "stage_replacement.target"),
            registry=registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "stage_model"
            package.mkdir()
            (package / "__init__.py").write_text(
                "from stage_vendor import FlashMHA\n"
                "from . import optional_extension\n"
                "from . import sparse\n"
                "def target(value): return value\n",
                encoding="utf-8",
            )
            (package / "sparse.py").write_text(
                "from stage_framework import CONV_LAYERS\n"
                "@CONV_LAYERS.register_module()\n"
                "class SparseConv3d: pass\n",
                encoding="utf-8",
            )
            (root / "stage_replacement.py").write_text(
                "def target(value): return value + 1\n", encoding="utf-8"
            )
            config = root / "optimization.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: stage, version: "1"}
                    optimization_groups:
                      - {id: stage.import_compatibility}
                      - {id: stage.runtime}
                    """
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, str(root))
            with mock.patch.dict(
                sys.modules,
                {
                    "stage_vendor": vendor,
                    "stage_framework": framework,
                },
                clear=False,
            ):
                try:
                    prepared = check(
                        optimization_config_path=config,
                        registry=registry,
                    )
                    self.assertEqual(
                        tuple(group.decision.value for group in prepared.groups),
                        ("apply", "apply"),
                    )
                    self.assertFalse(hasattr(vendor, "FlashMHA"))
                    self.assertNotIn("stage_model", sys.modules)
                    self.assertIs(
                        external_registry.module_dict["SparseConv3d"],
                        ExistingSparseConv3d,
                    )

                    with mock.patch("turbo_physai.engine._apply_called", False):
                        report = apply(
                            optimization_config_path=config,
                            registry=registry,
                        )
                    self.assertEqual(report.summary["applied"], 2)
                    model = importlib.import_module("stage_model")
                    self.assertEqual(model.target(3), 4)
                    self.assertIs(vendor.FlashMHA, MHA)
                    self.assertEqual(
                        external_registry.module_dict["SparseConv3d"].__module__,
                        "stage_model.sparse",
                    )
                finally:
                    sys.path.remove(str(root))
                    for name in tuple(sys.modules):
                        if name.startswith("stage_model") or name == "stage_replacement":
                            sys.modules.pop(name, None)
                    if hasattr(vendor, "FlashMHA"):
                        delattr(vendor, "FlashMHA")

    def test_compatibility_failure_blocks_model_import(self):
        vendor = types.ModuleType("blocked_vendor")
        registry = Registry()
        group(
            "blocked.import_compatibility",
            import_alias("blocked_vendor", "Missing", "Alias"),
            registry=registry,
        )
        from turbo_physai.engine.definitions import replace

        group(
            "blocked.runtime",
            replace("blocked_model.target", "blocked_replacement.target"),
            registry=registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "blocked_model.py").write_text(
                "raise AssertionError('model import must not run')\n",
                encoding="utf-8",
            )
            (root / "blocked_replacement.py").write_text(
                "def target(value): return value\n",
                encoding="utf-8",
            )
            config = root / "optimization.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: blocked, version: "1"}
                    optimization_groups:
                      - {id: blocked.import_compatibility}
                      - {id: blocked.runtime}
                    """
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, str(root))
            with mock.patch.dict(
                sys.modules, {"blocked_vendor": vendor}, clear=False
            ):
                try:
                    prepared = check(
                        optimization_config_path=config,
                        registry=registry,
                    )
                    self.assertEqual(
                        tuple(item.decision.value for item in prepared.groups),
                        ("block", "block"),
                    )
                    self.assertNotIn("blocked_model", sys.modules)
                finally:
                    sys.path.remove(str(root))
                    sys.modules.pop("blocked_model", None)
                    sys.modules.pop("blocked_replacement", None)

    def test_compatibility_group_failure_restores_earlier_members(self):
        vendor = types.ModuleType("rollback_vendor")

        class MHA:
            pass

        vendor.MHA = MHA
        framework = types.ModuleType("rollback_framework")
        external_registry = FakeRegistry()
        external_registry.module_dict["Unexpected"] = object()
        framework.CONV_LAYERS = external_registry
        registry = Registry()
        group(
            "rollback.import_compatibility",
            import_alias("rollback_vendor", "MHA", "FlashMHA"),
            registry_override(
                "rollback_model.sparse",
                "rollback_framework.CONV_LAYERS",
                names=("SparseConv3d",),
            ),
            registry=registry,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "rollback_model"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "sparse.py").write_text(
                "from rollback_framework import CONV_LAYERS\n"
                "@CONV_LAYERS.register_module()\n"
                "class Unexpected: pass\n",
                encoding="utf-8",
            )
            config = root / "optimization.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    schema_version: turbophysai/optimization-config/v1
                    kind: OptimizationConfig
                    metadata: {id: rollback, version: "1"}
                    optimization_groups:
                      - {id: rollback.import_compatibility}
                    """
                ),
                encoding="utf-8",
            )
            sys.path.insert(0, str(root))
            with mock.patch.dict(
                sys.modules,
                {
                    "rollback_vendor": vendor,
                    "rollback_framework": framework,
                },
                clear=False,
            ):
                try:
                    with mock.patch("turbo_physai.engine._apply_called", False):
                        report = apply(
                            optimization_config_path=config,
                            registry=registry,
                        )
                    self.assertEqual(report.summary["rolled_back"], 1)
                    self.assertFalse(hasattr(vendor, "FlashMHA"))
                    self.assertNotIn("rollback_model.sparse", sys.modules)
                finally:
                    sys.path.remove(str(root))
                    for name in tuple(sys.modules):
                        if name.startswith("rollback_model"):
                            sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
