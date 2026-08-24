# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import Registry, default_registry
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.optimizations.common.mmcv import catalog


class BuiltinMsdaTest(unittest.TestCase):
    def test_catalog_declares_public_mmcv_ext_contract(self):
        group = default_registry.get_group(catalog.MSDA.group_id)
        self.assertEqual(group, catalog.MSDA.definition)
        self.assertEqual(catalog.MSDA.group_id, "mmcv.msda")
        self.assertEqual(len(catalog.MSDA.specs), 2)
        self.assertEqual(
            tuple(spec.target for spec in catalog.MSDA.specs),
            (
                "mmcv._ext.ms_deform_attn_forward",
                "mmcv._ext.ms_deform_attn_backward",
            ),
        )
        self.assertTrue(
            all(
                spec.mechanism is Mechanism.REPLACE
                for spec in catalog.MSDA.specs
            )
        )

    def test_public_group_is_selected_by_mmcv_config(self):
        config_path = (
            Path(__file__).parents[2]
            / "turbo_physai"
            / "optimizations"
            / "common"
            / "mmcv"
            / "configs"
            / "optimization.yaml"
        )
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [entry["id"] for entry in raw["optimization_groups"]],
            ["mmcv.msda"],
        )

    def test_target_pair_applies_and_restores_together(self):
        def original_forward(*args, **kwargs):
            return args, kwargs

        def original_backward(*args, **kwargs):
            return args, kwargs

        def replacement_forward(*args, **kwargs):
            return args, kwargs

        def replacement_backward(*args, **kwargs):
            return args, kwargs

        mmcv = types.ModuleType("mmcv")
        ext = types.ModuleType("mmcv._ext")
        ext.ms_deform_attn_forward = original_forward
        ext.ms_deform_attn_backward = original_backward
        mmcv._ext = ext
        replacement = types.ModuleType(
            "turbo_physai.optimizations.common.mmcv.msda"
        )
        replacement.ms_deform_attn_forward = replacement_forward
        replacement.ms_deform_attn_backward = replacement_backward
        modules = {
            "mmcv": mmcv,
            "mmcv._ext": ext,
            replacement.__name__: replacement,
        }
        registry = Registry()
        catalog.MSDA.register(registry)
        handler = default_handlers()[Mechanism.REPLACE]
        snapshots = []

        with mock.patch.dict(sys.modules, modules):
            for replacement_id in registry.get_group("mmcv.msda").members:
                prepared = handler.prepare(registry.get_spec(replacement_id), {})
                snapshots.append(handler.snapshot(prepared))
                handler.apply(prepared)
            self.assertIs(ext.ms_deform_attn_forward, replacement_forward)
            self.assertIs(ext.ms_deform_attn_backward, replacement_backward)
            for snapshot in reversed(snapshots):
                handler.restore(snapshot)

        self.assertIs(ext.ms_deform_attn_forward, original_forward)
        self.assertIs(ext.ms_deform_attn_backward, original_backward)

    def test_public_replacement_only_delegates_to_lightop(self):
        script = r'''
import sys
import types

lightop = types.ModuleType("lightop")
calls = []

def forward(value, shapes, starts, locations, weights, step):
    calls.append(("forward", value, shapes, starts, locations, weights, step))
    return "forward-result"

def backward(value, shapes, starts, locations, weights, grad_output,
             grad_value, grad_locations, grad_weights, step):
    calls.append((
        "backward", value, shapes, starts, locations, weights, grad_output,
        grad_value, grad_locations, grad_weights, step,
    ))

class NonContiguousGradient:
    def is_contiguous(self):
        return False

    def contiguous(self):
        return "contiguous-gradient"

lightop.op = types.SimpleNamespace(
    ms_deform_attn_forward=forward,
    ms_deform_attn_backward=backward,
)
sys.modules["lightop"] = lightop

from turbo_physai.optimizations.common.mmcv import msda
assert "torch" not in sys.modules

value = object()
shapes = object()
starts = object()
locations = object()
weights = object()
args = (value, shapes, starts, locations, weights, 64)

output = msda.ms_deform_attn_forward(*args)
assert output == "forward-result"
assert calls[0] == ("forward", *args)

grad_value = object()
grad_locations = object()
grad_weights = object()
result = msda.ms_deform_attn_backward(
    value, shapes, starts, locations, weights, NonContiguousGradient(),
    grad_value, grad_locations, grad_weights, 64,
)
assert result is None
assert calls[1] == (
    "backward", value, shapes, starts, locations, weights,
    "contiguous-gradient", grad_value, grad_locations, grad_weights, 64,
)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
