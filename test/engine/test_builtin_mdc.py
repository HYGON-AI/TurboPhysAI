# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
import types
import unittest
from unittest import mock

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import default_registry
from turbo_physai.optimizations.common.mmcv import catalog
from turbo_physai.optimizations.common.mmcv import modulated_deform_conv


def _fake_torch():
    torch_module = types.ModuleType("torch")
    torch_module.float16 = object()
    torch_module.float32 = object()

    class Tensor:
        def __init__(self, shape, dtype=None, is_cuda=True):
            self.shape = shape
            self.dtype = dtype or torch_module.float32
            self.is_cuda = is_cuda

        def dim(self):
            return len(self.shape)

        def numel(self):
            return math.prod(self.shape)

    torch_module.Tensor = Tensor
    return torch_module, Tensor


class BuiltinMdcTest(unittest.TestCase):
    def test_catalog_declares_public_mmcv_contract(self):
        group = default_registry.get_group(catalog.MDC.group_id)
        self.assertEqual(group, catalog.MDC.definition)
        self.assertEqual(catalog.MDC.group_id, "mmcv.mdc")
        self.assertEqual(len(catalog.MDC.specs), 1)
        spec = catalog.MDC.specs[0]
        self.assertEqual(
            spec.target,
            "mmcv.ops.modulated_deform_conv.modulated_deform_conv2d",
        )
        self.assertEqual(spec.aliases, ("mmcv.ops.modulated_deform_conv2d",))
        self.assertIs(spec.mechanism, Mechanism.REPLACE)
        self.assertEqual(
            spec.runtime_condition,
            "turbo_physai.optimizations.common.mmcv.modulated_deform_conv."
            "is_supported_mdc_call",
        )

    def test_runtime_condition_accepts_standard_supported_arguments(self):
        torch_module, Tensor = _fake_torch()
        input_tensor = Tensor((1, 4, 5, 6))
        weight = Tensor((4, 4, 3, 3))
        bias = Tensor((4,))
        offset = Tensor((1, 36, 5, 6))
        mask = Tensor((1, 18, 5, 6))

        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            self.assertTrue(
                modulated_deform_conv.is_supported_mdc_call(
                    input_tensor,
                    offset,
                    mask,
                    weight,
                    bias,
                    1,
                    1,
                    1,
                    1,
                    2,
                )
            )

    def test_runtime_condition_rejects_unsupported_or_invalid_arguments(self):
        torch_module, Tensor = _fake_torch()
        input_tensor = Tensor((1, 4, 5, 6))
        weight = Tensor((4, 4, 3, 3))
        offset = Tensor((1, 18, 5, 6))
        mask = Tensor((1, 9, 5, 6))

        with mock.patch.dict(sys.modules, {"torch": torch_module}):
            self.assertFalse(
                modulated_deform_conv.is_supported_mdc_call(
                    input_tensor,
                    offset,
                    mask,
                    weight,
                    None,
                    groups=2,
                )
            )
            self.assertFalse(
                modulated_deform_conv.is_supported_mdc_call(
                    input_tensor,
                    Tensor((1, 20, 5, 6)),
                    mask,
                    weight,
                )
            )
            self.assertFalse(
                modulated_deform_conv.is_supported_mdc_call(
                    Tensor((1, 4, 5, 6), is_cuda=False),
                    offset,
                    mask,
                    weight,
                )
            )

    def test_replacement_loads_operator_lazily_and_delegates(self):
        calls = []
        operator_module = types.ModuleType(
            "turbo_physai.operators.modulated_deform_conv"
        )

        def operator(*args, **kwargs):
            calls.append((args, kwargs))
            return "operator-result"

        operator_module.modulated_deform_conv2d = operator
        with mock.patch.dict(sys.modules, {operator_module.__name__: operator_module}):
            result = modulated_deform_conv.modulated_deform_conv2d("input", padding=1)

        self.assertEqual(result, "operator-result")
        self.assertEqual(calls, [(("input",), {"padding": 1})])


if __name__ == "__main__":
    unittest.main()
