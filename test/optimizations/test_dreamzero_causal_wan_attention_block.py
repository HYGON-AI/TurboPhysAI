# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Modulation/FFN numerical equivalence and conditional I2V compilation.

FP64 CPU tensors, fixed seed/weights/upstream gradients; tolerance 1e-12.
Frequency/cache metadata is forwarded unchanged and needs no gradient.
"""

from __future__ import annotations

import copy
import sys
import types
from contextlib import contextmanager
from unittest import mock

import pytest


torch = pytest.importorskip("torch")
from torch import nn

from turbo_physai.optimizations.models.dreamzero import (
    CausalWanAttentionBlock as optimized,
)


@contextmanager
def _fake_module(name, **attributes):
    """Temporarily install an isolated module import; restore it on exit."""
    parts = name.split(".")
    modules = {}
    for index in range(1, len(parts) + 1):
        path = ".".join(parts[:index])
        module = types.ModuleType(path)
        module.__path__ = []
        modules[path] = module
        if index > 1:
            setattr(modules[".".join(parts[:index - 1])], parts[index - 1], module)
    target = modules[name]
    vars(target).update(attributes)
    with mock.patch.dict(sys.modules, modules):
        yield target


def assert_contract(actual, expected):
    assert isinstance(actual, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device


def assert_parameter_grads(actual_model, expected_model, *, rtol=1e-5, atol=1e-6):
    actual_params = dict(actual_model.named_parameters())
    expected_params = dict(expected_model.named_parameters())
    assert actual_params.keys() == expected_params.keys()
    for name, parameter in expected_params.items():
        got = actual_params[name].grad
        if parameter.grad is None:
            assert got is None, name
        else:
            assert got is not None, name
            torch.testing.assert_close(got, parameter.grad, rtol=rtol, atol=atol)


class SelfAttention(nn.Linear):
    def forward(self, x, **kwargs):
        self.arguments = kwargs
        return super().forward(x), kwargs["kv_cache"]


class CrossAttention(nn.Linear):
    def forward(self, x, context):
        return super().forward(x) + context.mean(1, keepdim=True)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(8) for _ in range(3))
        self.self_attn = SelfAttention(8, 8)
        self.cross_attn = CrossAttention(8, 8)
        self.ffn = nn.Sequential(nn.Linear(8, 12), nn.GELU(), nn.Linear(12, 8))
        self.modulation = nn.Parameter(torch.randn(1, 6, 8))


def reference_forward(self, x, e, context):
    # Baseline aligns each of six modulation chunks independently.
    parts = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
    aligned = []
    for part in parts:
        if part.shape[1] < x.shape[1]:
            count = (x.shape[1] + part.shape[1] - 1) // part.shape[1]
            part = part.repeat_interleave(count, dim=1)
        aligned.append(part[:, :x.shape[1]].squeeze(2))
    shift, scale, gate, ffn_shift, ffn_scale, ffn_gate = aligned
    y = nn.functional.linear(
        self.norm1(x) * (1 + scale) + shift,
        self.self_attn.weight, self.self_attn.bias,
    )
    x = x + gate * y
    x = x + self.cross_attn(self.norm3(x), context)
    return x + ffn_gate * self.ffn(self.norm2(x) * (1 + ffn_scale) + ffn_shift)


@pytest.mark.parametrize("with_cache", [False, True])
@pytest.mark.parametrize("modulation_length", [1, 3, 5, 8])
def test_alignment_outputs_all_training_gradients_and_cache_contract(
    modulation_length, with_cache
):
    torch.manual_seed(31)
    ref = Block().double()
    opt = copy.deepcopy(ref)
    inputs = [
        torch.randn(2, 5, 8, dtype=torch.float64),
        torch.randn(2, modulation_length, 6, 8, dtype=torch.float64),
        torch.randn(2, 4, 8, dtype=torch.float64),
    ]
    ref_inputs = [x.requires_grad_() for x in inputs]
    opt_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    want = reference_forward(ref, *ref_inputs)
    cache = torch.randn(2, 2, 1, 2, 4) if with_cache else None
    freqs, action_freqs, state_freqs = object(), object(), object()
    result = optimized.forward(
        opt, opt_inputs[0], opt_inputs[1], freqs, action_freqs,
        state_freqs, 4, opt_inputs[2], kv_cache=cache,
        current_start_frame=3, is_tf=not with_cache,
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    got, updated_cache = result
    assert updated_cache is cache
    assert_contract(got, want)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert opt.self_attn.arguments == dict(
        freqs=freqs, freqs_action=action_freqs, freqs_state=state_freqs,
        action_register_length=4, kv_cache=cache,
        is_tf=not with_cache, current_start_frame=3,
    )
    upstream = torch.randn_like(want)
    got.backward(upstream)
    want.backward(upstream)
    for actual, expected in zip(opt_inputs, ref_inputs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)
    assert_parameter_grads(opt, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("is_i2v", [False, True], ids=["other", "i2v"])
@pytest.mark.parametrize(
    "options", [None, {}, {"dynamic": False, "backend": "eager"}],
    ids=["none", "empty", "present"],
)
def test_only_i2v_target_compiled_after_original_init_with_options(options, is_i2v, monkeypatch):
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)

    class I2V(nn.Module):
        pass

    cross_attn = I2V() if is_i2v else nn.Identity()
    cross_attn.compile = mock.Mock()

    def original(owner, dim, *, heads):
        owner.cross_attn = cross_attn

    init = mock.Mock(side_effect=original)
    owner = types.SimpleNamespace()
    with _fake_module(
        "groot.vla.model.dreamzero.modules.wan2_1_submodule", WanI2VCrossAttention=I2V,
    ):
        wrapped = optimized.compile_cross_attn(init, options)
        assert wrapped(owner, 8, heads=2) is None
    init.assert_called_once_with(owner, 8, heads=2)
    assert wrapped.__wrapped__ is init
    if is_i2v:
        cross_attn.compile.assert_called_once_with(
            **({"fullgraph": True, "dynamic": True} | (options or {}))
        )
    else:
        cross_attn.compile.assert_not_called()
    assert optimized._COMPILE_DEFAULTS == {"fullgraph": True, "dynamic": True}
