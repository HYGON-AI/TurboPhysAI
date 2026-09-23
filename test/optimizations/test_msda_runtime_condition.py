# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Check MSDA routing with real tensor metadata and isolated backend spies."""

import sys
import types
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.optimizations.common.mmcv import catalog


def inputs(device="cpu", dtype=torch.float32):
    return dict(
        value=torch.rand(2, 8, 2, 4, device=device, dtype=dtype),
        value_spatial_shapes=torch.tensor([[2, 2], [2, 2]], device=device),
        value_level_start_index=torch.tensor([0, 4], device=device),
        sampling_locations=torch.rand(2, 3, 2, 2, 2, 2, device=device, dtype=dtype),
        attention_weights=torch.rand(2, 3, 2, 2, 2, device=device, dtype=dtype),
        im2col_step=64,
    )


def backward_inputs(args):
    value = args["value"]
    return dict(
        args,
        grad_output=value.new_ones(2, 3, 8),
        grad_value=torch.zeros_like(value),
        grad_sampling_loc=torch.zeros_like(args["sampling_locations"]),
        grad_attn_weight=torch.zeros_like(args["attention_weights"]),
    )


@pytest.fixture
def hcu_inputs():
    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires HCU")
    return inputs("cuda")


def assert_dispatch(direction, args, supported):
    """Apply the public catalog declaration, call it, then restore its target."""
    spec = catalog.MSDA.specs[0 if direction == "forward" else 1]
    name = "ms_deform_attn_" + direction
    original = mock.Mock(return_value="original")
    optimized = mock.Mock(return_value="lightop")
    mmcv = types.ModuleType("mmcv")
    ext = types.ModuleType("mmcv._ext")
    mmcv._ext = ext
    setattr(ext, name, original)
    replacement = types.ModuleType("turbo_physai.operators.multi_scale_deformable_attention")
    setattr(replacement, name, optimized)
    handler = default_handlers()[Mechanism.REPLACE]
    with mock.patch.dict(sys.modules, {
        "mmcv": mmcv, "mmcv._ext": ext, replacement.__name__: replacement,
    }):
        prepared = handler.prepare(spec, {})
        snapshot = handler.snapshot(prepared)
        try:
            handler.apply(prepared)
            result = getattr(ext, name)(**args)
            assert result == ("lightop" if supported else "original")
            selected, unused = (optimized, original) if supported else (original, optimized)
            selected.assert_called_once_with(**args)
            unused.assert_not_called()
        finally:
            handler.restore(snapshot)
        assert getattr(ext, name) is original


def test_cpu_calls_use_original():
    args = inputs()
    assert_dispatch("forward", args, False)
    assert_dispatch("backward", backward_inputs(args), False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_supported_calls_use_lightop(hcu_inputs, dtype):
    args = inputs("cuda", dtype)
    assert_dispatch("forward", args, True)
    assert_dispatch("backward", backward_inputs(args), True)


@pytest.mark.parametrize("case", [
    "fp16", "bf16", "mixed_dtype", "cpu_index", "int32_index",
    "noncontiguous", "wrong_rank", "wrong_levels", "wrong_weights",
    "zero_step", "negative_step", "bool_step", "uneven_step", "empty_batch",
])
def test_unsupported_inputs_fall_back_in_both_directions(hcu_inputs, case):
    args = hcu_inputs
    if case in ("fp16", "bf16"):
        dtype = torch.float16 if case == "fp16" else torch.bfloat16
        for key in ("value", "sampling_locations", "attention_weights"):
            args[key] = args[key].to(dtype)
    elif case == "mixed_dtype":
        args["attention_weights"] = args["attention_weights"].double()
    elif case == "cpu_index":
        args["value_level_start_index"] = args["value_level_start_index"].cpu()
    elif case == "int32_index":
        args["value_spatial_shapes"] = args["value_spatial_shapes"].int()
    elif case == "noncontiguous":
        args["sampling_locations"] = args["sampling_locations"].transpose(-1, -2)
    elif case == "wrong_rank":
        args["value"] = args["value"].unsqueeze(0)
    elif case == "wrong_levels":
        args["value_level_start_index"] = args["value_level_start_index"][:1]
    elif case == "wrong_weights":
        args["attention_weights"] = args["attention_weights"][:, :1].contiguous()
    elif case == "empty_batch":
        args["value"] = args["value"][:0]
    else:
        args["im2col_step"] = {
            "zero_step": 0, "negative_step": -1, "bool_step": True, "uneven_step": 3,
        }[case]
        if case == "uneven_step":
            args["value"] = args["value"].repeat(2, 1, 1, 1)
            args["sampling_locations"] = args["sampling_locations"].repeat(2, 1, 1, 1, 1, 1)
            args["attention_weights"] = args["attention_weights"].repeat(2, 1, 1, 1, 1)
    assert_dispatch("forward", args, False)
    assert_dispatch("backward", backward_inputs(args), False)


@pytest.mark.parametrize("case", ["dtype", "shape", "device", "buffer_layout"])
def test_unsupported_gradient_buffers_use_original(hcu_inputs, case):
    args = backward_inputs(hcu_inputs)
    if case == "dtype":
        args["grad_output"] = args["grad_output"].double()
    elif case == "shape":
        args["grad_value"] = args["grad_value"][:1]
    elif case == "device":
        args["grad_attn_weight"] = args["grad_attn_weight"].cpu()
    else:
        args["grad_sampling_loc"] = args["grad_sampling_loc"].transpose(-1, -2)
    assert_dispatch("backward", args, False)


def test_noncontiguous_output_gradient_is_supported(hcu_inputs):
    args = backward_inputs(hcu_inputs)
    args["grad_output"] = args["value"].new_ones(2, 8, 3).transpose(1, 2)
    assert not args["grad_output"].is_contiguous()
    assert_dispatch("backward", args, True)


def test_non_hip_build_uses_original(hcu_inputs, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None)
    assert_dispatch("forward", hcu_inputs, False)
    assert_dispatch("backward", backward_inputs(hcu_inputs), False)
