# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import turbo_physai
import turbo_physai.engine as engine_module
from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import Registry, default_registry
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.engine.execution.replacements.base import resolve_replacement
from turbo_physai.engine.execution.replacements.replace import ReplaceHandler
from turbo_physai.engine.config.loader import load_optimization_config
from turbo_physai.optimizations.models.bevformer import catalog
from turbo_physai.optimizations.common.mmcv import catalog as mmcv_catalog


BEVFORMER_GROUPS = (
    catalog.MDC,
    catalog.MSDA,
    catalog.GEOMETRY,
    catalog.TSA,
    catalog.GRID_MASK,
    catalog.COMPILE,
    catalog.TRAINING,
)
FP16_SPEC, FP32_SPEC = catalog.MSDA.specs


class OriginalFp16:
    pass


class OriginalFp32:
    pass


class ReplacementFp16:
    pass


class ReplacementFp32:
    pass


class BevFormerMsdaOptimizationEngineTest(unittest.TestCase):
    def setUp(self):
        engine_module._apply_called = False

    def test_catalog_import_is_runtime_free_and_default_registry_is_narrow(self):
        self.assertNotIn("lightop", sys.modules)
        group = default_registry.get_group(catalog.MSDA.group_id)
        self.assertIsNotNone(group)
        self.assertTrue(
            ({optimization.group_id for optimization in BEVFORMER_GROUPS}
             | {mmcv_catalog.MSDA.group_id}).issubset(default_registry.groups),
        )
        self.assertEqual(
            group.members, (FP16_SPEC.replacement_id, FP32_SPEC.replacement_id)
        )

    def test_specs_declare_targets_aliases_and_replacements(self):
        fp16 = default_registry.get_spec(FP16_SPEC.replacement_id)
        fp32 = default_registry.get_spec(FP32_SPEC.replacement_id)
        self.assertEqual(fp16.mechanism, Mechanism.REPLACE)
        self.assertEqual(fp16.target, catalog.FP16_TARGET)
        self.assertEqual(fp16.aliases, catalog.FP16_ALIASES)
        self.assertEqual(
            fp16.replacement,
            "turbo_physai.optimizations.models.bevformer.msda."
            "MultiScaleDeformableAttnFunction_fp16",
        )
        self.assertEqual(fp32.target, catalog.FP32_TARGET)
        self.assertEqual(fp32.aliases, catalog.FP32_ALIASES)
        self.assertEqual(
            fp32.replacement,
            "turbo_physai.optimizations.models.bevformer.msda."
            "MultiScaleDeformableAttnFunction_fp32",
        )

    def test_catalog_generates_internal_specs_from_group_declarations(self):
        mdc_group = default_registry.get_group(catalog.MDC.group_id)
        self.assertEqual(mdc_group, catalog.MDC.definition)
        self.assertEqual(
            tuple(default_registry.get_spec(member) for member in mdc_group.members),
            catalog.MDC.specs,
        )
        self.assertEqual(catalog.MDC.specs[-1].mechanism, Mechanism.WRAPPER)

    def test_replacement_paths_resolve_directly(self):
        fake_module = types.ModuleType(
            "turbo_physai.optimizations.models.bevformer.msda"
        )
        fake_module.MultiScaleDeformableAttnFunction_fp16 = ReplacementFp16
        fake_module.MultiScaleDeformableAttnFunction_fp32 = ReplacementFp32
        with mock.patch.dict(sys.modules, {fake_module.__name__: fake_module}):
            fp16 = resolve_replacement(
                default_registry.get_spec(FP16_SPEC.replacement_id).replacement
            )
            fp32 = resolve_replacement(
                default_registry.get_spec(FP32_SPEC.replacement_id).replacement
            )
        self.assertIs(fp16, ReplacementFp16)
        self.assertIs(fp32, ReplacementFp32)

    def test_missing_torch_custom_op_fails_when_replacement_is_resolved(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.library = types.SimpleNamespace(
            custom_op=None, register_autograd=lambda *args, **kwargs: None
        )
        fake_torch.autograd = types.SimpleNamespace(Function=object)
        fake_lightop = types.ModuleType("lightop")
        fake_lightop.op = types.SimpleNamespace(
            ms_deform_attn_forward=lambda *args: None,
            ms_deform_attn_backward=lambda *args: None,
        )
        module_name = "turbo_physai.optimizations.models.bevformer.msda"
        saved = sys.modules.pop(module_name, None)
        try:
            with mock.patch.dict(
                sys.modules, {"torch": fake_torch, "lightop": fake_lightop}
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "torch.library.custom_op"
                ):
                    resolve_replacement(
                        default_registry.get_spec(FP16_SPEC.replacement_id).replacement
                    )
        finally:
            sys.modules.pop(module_name, None)
            if saved is not None:
                sys.modules[module_name] = saved

    def test_real_torch_fake_lightop_forward_backward_fake_and_compile(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("Torch is not installed in the optimization engine test interpreter")
        script = r'''
import inspect
import sys
import traceback
import types
import torch

lightop = types.ModuleType("lightop")
def forward(value, shapes, starts, locations, weights, step):
    queries = locations.shape[1]
    return value[:, :queries].reshape(value.shape[0], queries, -1).clone()
def backward(value, shapes, starts, locations, weights, grad_output,
             grad_value, grad_locations, grad_weights, step):
    grad_value.fill_(1)
    grad_locations.fill_(2)
    grad_weights.fill_(3)
lightop.op = types.SimpleNamespace(
    ms_deform_attn_forward=forward,
    ms_deform_attn_backward=backward,
)
sys.modules["lightop"] = lightop

from turbo_physai.optimizations.models.bevformer import msda

class Original(torch.autograd.Function):
    pass
assert inspect.signature(Original) == inspect.signature(
    msda.MultiScaleDeformableAttnFunction_fp16
)
value = torch.randn(1, 2, 1, 2, requires_grad=True)
shapes = torch.tensor([[1, 2]], dtype=torch.long)
starts = torch.tensor([0], dtype=torch.long)
locations = torch.randn(1, 2, 1, 1, 1, 2, requires_grad=True)
weights = torch.randn(1, 2, 1, 1, 1, requires_grad=True)
args = (value, shapes, starts, locations, weights, 64)
output = msda.MultiScaleDeformableAttnFunction_fp32.apply(*args)
assert output.shape == (1, 2, 2)
output.sum().backward()
assert torch.equal(value.grad, torch.ones_like(value))
assert torch.equal(locations.grad, torch.full_like(locations, 2))
assert torch.equal(weights.grad, torch.full_like(weights, 3))

with torch._subclasses.fake_tensor.FakeTensorMode() as mode:
    fake_args = tuple(mode.from_tensor(item) if isinstance(item, torch.Tensor) else item for item in args)
    fake_output = msda.MultiScaleDeformableAttnFunction_fp32.apply(*fake_args)
    assert fake_output.shape == (1, 2, 2)

compiled = torch.compile(
    msda.MultiScaleDeformableAttnFunction_fp32.apply,
    backend="eager",
    fullgraph=True,
)
assert compiled(*tuple(item.detach() if isinstance(item, torch.Tensor) else item for item in args)).shape == (1, 2, 2)

def injected_failure(*args):
    raise RuntimeError("injected LightOp failure")
msda._msda_forward = injected_failure
try:
    msda.MultiScaleDeformableAttnFunction_fp32.apply(
        *tuple(item.detach() if isinstance(item, torch.Tensor) else item for item in args)
    )
except RuntimeError:
    assert (
        "turbo_physai/optimizations/models/bevformer/msda.py"
        in traceback.format_exc()
    )
else:
    raise AssertionError("LightOp failure was unexpectedly swallowed")
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if "torch.library.custom_op" in completed.stderr:
            self.skipTest("installed Torch does not provide torch.library.custom_op")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_targets_and_aliases_apply_and_restore_together(self):
        modules = self._fake_model_modules()
        modules.update(self._fake_replacement_modules())
        registry = Registry()
        catalog.MSDA.register(registry)
        handler = default_handlers()[Mechanism.REPLACE]
        snapshots = []
        with mock.patch.dict(sys.modules, modules):
            for replacement_id in (FP16_SPEC.replacement_id, FP32_SPEC.replacement_id):
                spec = registry.get_spec(replacement_id)
                prepared = handler.prepare(spec, {})
                snapshots.append(handler.snapshot(prepared))
                changed = handler.apply(prepared)
                self.assertEqual(changed, (spec.target,) + spec.aliases)
            self._assert_model_identity(modules, ReplacementFp16, ReplacementFp32)
            for snapshot in reversed(snapshots):
                handler.restore(snapshot)
            self._assert_model_identity(modules, OriginalFp16, OriginalFp32)

    def test_second_spec_failure_rolls_back_entire_group(self):
        modules = self._fake_model_modules()
        modules.update(self._fake_replacement_modules())
        registry = Registry()
        catalog.MSDA.register(registry)
        original_apply = ReplaceHandler.apply

        def fail_fp32(handler, prepared):
            if prepared.spec.replacement_id == FP32_SPEC.replacement_id:
                raise RuntimeError("injected fp32 apply failure")
            return original_apply(handler, prepared)

        config = textwrap.dedent(
            f"""
            schema_version: turbophysai/optimization-config/v1
            kind: OptimizationConfig
            metadata: {{id: rollback, version: "1"}}
            optimization_groups:
              - id: {catalog.MSDA.group_id}
            """
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimization_config_path = root / "config.yaml"
            optimization_config_path.write_text(config, encoding="utf-8")
            with mock.patch.dict(sys.modules, modules), mock.patch.object(
                ReplaceHandler, "apply", fail_fp32
            ):
                report = turbo_physai.apply(
                    optimization_config_path=optimization_config_path,
                    registry=registry,
                )
            self.assertEqual(report.execution[0].status.value, "rolled_back")
        self._assert_model_identity(modules, OriginalFp16, OriginalFp32)

    def test_complete_optimization_config_preserves_declared_group_order_and_trust(self):
        path = (
            Path(turbo_physai.__file__).parent
            / "optimizations"
            / "models"
            / "bevformer"
            / "configs"
            / "optimization.yaml"
        )
        config = load_optimization_config(path)
        self.assertEqual(config.metadata.id, "model.bevformer.base.hcu")
        self.assertEqual(
            config.compatibility["commits"],
            ("66b65f3a1f58caf0507cb2a971b9c0e7f842376c",),
        )
        self.assertEqual(
            tuple(entry.id for entry in config.optimization_groups),
            tuple(optimization.group_id for optimization in BEVFORMER_GROUPS),
        )
        self.assertTrue(all(entry.enabled for entry in config.optimization_groups))
        for entry in config.optimization_groups:
            group = default_registry.get_group(entry.id)
            self.assertIsNotNone(group)
            for replacement_id in group.members:
                target = default_registry.get_spec(replacement_id).target
                self.assertIn(target, entry.trust["source_hashes"])
                self.assertIn(target, entry.trust["ast_hashes"])

        msda = next(
            entry
            for entry in config.optimization_groups
            if entry.id == catalog.MSDA.group_id
        )
        for spec_id in (FP16_SPEC.replacement_id, FP32_SPEC.replacement_id):
            target = default_registry.get_spec(spec_id).target
            self.assertTrue(
                msda.trust["source_hashes"][target][0].startswith("source-v1:")
            )
            self.assertTrue(
                msda.trust["ast_hashes"][target][0].startswith("ast-v1:")
            )

        geometry = config.optimization_groups[2]
        self.assertEqual(geometry.options, {})

    @staticmethod
    def _fake_model_modules():
        prefix = "projects.mmdet3d_plugin.bevformer.modules"
        definitions = {
            "multi_scale_deformable_attn_function": (OriginalFp16, OriginalFp32),
            "spatial_cross_attention": (OriginalFp16, OriginalFp32),
            "temporal_self_attention": (None, OriginalFp32),
            "decoder": (OriginalFp16, OriginalFp32),
        }
        modules = {}
        for suffix, (fp16, fp32) in definitions.items():
            module = types.ModuleType(f"{prefix}.{suffix}")
            if fp16 is not None:
                module.MultiScaleDeformableAttnFunction_fp16 = fp16
            if fp32 is not None:
                module.MultiScaleDeformableAttnFunction_fp32 = fp32
            modules[module.__name__] = module
        return modules

    @staticmethod
    def _fake_replacement_modules():
        module = types.ModuleType(
            "turbo_physai.optimizations.models.bevformer.msda"
        )
        module.MultiScaleDeformableAttnFunction_fp16 = ReplacementFp16
        module.MultiScaleDeformableAttnFunction_fp32 = ReplacementFp32
        return {module.__name__: module}

    def _assert_model_identity(self, modules, fp16, fp32):
        for path in (catalog.FP16_TARGET,) + catalog.FP16_ALIASES:
            module_name, attribute = path.rsplit(".", 1)
            self.assertIs(getattr(modules[module_name], attribute), fp16)
        for path in (catalog.FP32_TARGET,) + catalog.FP32_ALIASES:
            module_name, attribute = path.rsplit(".", 1)
            self.assertIs(getattr(modules[module_name], attribute), fp32)


if __name__ == "__main__":
    unittest.main()
