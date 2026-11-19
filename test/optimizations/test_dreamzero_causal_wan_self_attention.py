# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Self-attention numerical comparisons; full forward uses the optional model baseline."""

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
    CausalWanSelfAttention as optimized,
)
from turbo_physai.optimizations.models.dreamzero.real_rope import rope_params_real

try:
    from groot.vla.model.dreamzero.modules import wan_video_dit_action_casual_chunk as baseline
except ModuleNotFoundError as exc:
    if exc.name != "groot":
        raise
    baseline = None

requires_baseline = pytest.mark.skipif(
    baseline is None, reason="requires the DreamZero model repository"
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


class ExactAttention(nn.Module):
    """Independent attention with FA2's bottom-right causal alignment."""

    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, q, k, v):
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
        if self.causal:
            mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril(
                diagonal=k.shape[-2] - q.shape[-2]
            )
            scores = scores.masked_fill(~mask, -torch.inf)
        return (scores.softmax(-1) @ v).transpose(1, 2)


def _models(local_attn_size):
    torch.manual_seed(5)
    reference = baseline.CausalWanSelfAttention(
        dim=24, num_heads=2, frame_seqlen=2, num_frame_per_block=2,
        num_action_per_block=2, num_state_per_block=2,
        local_attn_size=local_attn_size,
    ).double()
    reference.attn = ExactAttention()
    reference.causal_attn = ExactAttention(causal=True)
    candidate = copy.deepcopy(reference)
    candidate.forward = types.MethodType(optimized.forward, candidate)
    candidate._process_state_blocks = types.MethodType(optimized.process_state_blocks, candidate)
    return reference, candidate


def _compare(reference, candidate, length, image_length, registers, is_tf, cache=False):
    table = rope_params_real(64, 12)
    complex_table = torch.view_as_complex(table.reshape(64, 6, 2))
    x = torch.randn(2, length, 24, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    kwargs = dict(action_register_length=registers, is_tf=is_tf)
    if cache:
        kwargs.update(
            kv_cache=torch.randn(2, 2, 3, 2, 12, dtype=torch.float64),
            current_start_frame=3,
        )
    want, want_cache = reference(
        x, complex_table[:image_length, None], complex_table, complex_table, **kwargs,
    )
    got, got_cache = candidate(y, table[:image_length, None], table, table, **kwargs)
    assert_contract(got, want)
    torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6)
    if cache:
        torch.testing.assert_close(got_cache, want_cache, rtol=2e-6, atol=2e-6)
    else:
        assert got_cache is None
    upstream = torch.randn_like(got)
    want.backward(upstream)
    got.backward(upstream)
    torch.testing.assert_close(y.grad, x.grad, rtol=2e-5, atol=2e-5)
    for (name, expected), (_, actual) in zip(reference.named_parameters(), candidate.named_parameters()):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-5)


@requires_baseline
@pytest.mark.parametrize("local_attn_size", [-1, 3])
def test_teacher_forcing_multiple_blocks_with_shared_image_action_kv(local_attn_size):
    _compare(*_models(local_attn_size), 28, 10, 8, True)


@requires_baseline
def test_image_only_teacher_forcing():
    _compare(*_models(-1), 20, 10, None, True)


@requires_baseline
def test_non_teacher_forcing():
    _compare(*_models(-1), 18, 10, 8, False)


@requires_baseline
@pytest.mark.parametrize("registers", [None, 4])
def test_kv_cache_with_and_without_registers(registers):
    _compare(*_models(-1), 4 + (registers or 0), 4, registers, False, cache=True)


@requires_baseline
def test_folded_state_blocks_match_loop_and_reduce_calls():
    reference, candidate = _models(-1)
    q, k, v = (torch.randn(2, 6, 2, 12, dtype=torch.float64) for _ in range(3))
    with mock.patch.object(candidate.attn, "forward", wraps=candidate.attn.forward) as calls:
        got = candidate._process_state_blocks(q, k, v, 6)
        assert calls.call_count == 1
    torch.testing.assert_close(got, reference._process_state_blocks(q, k, v, 6))
    empty = q[:, :0]
    assert candidate._process_state_blocks(empty, k[:, :0], v[:, :0], 0) is empty
    with pytest.raises(ValueError):
        candidate._process_state_blocks(q[:, :5], k[:, :5], v[:, :5], 5)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("blocks", [1, 3])
def test_block_folding_matches_loop_outputs_and_all_qkv_gradients(blocks, strided):
    """Synthetic FP64 attention; forward/all QKV gradients use 1e-12."""
    torch.manual_seed(79)
    values = [torch.randn(2, blocks * 2, 2, 8, dtype=torch.float64)[..., ::2] for _ in range(3)]
    if not strided:
        values = [value.contiguous() for value in values]
    values = [value.requires_grad_() for value in values]
    refs = [value.detach().clone().requires_grad_() for value in values]
    attention = mock.Mock(side_effect=ExactAttention())
    owner = types.SimpleNamespace(num_state_per_block=2, attn=attention)
    got = optimized.process_state_blocks(owner, *values, blocks * 2)
    want = torch.cat(
        [ExactAttention()(*(value[:, start:start + 2] for value in refs)) for start in range(0, blocks * 2, 2)], 1,
    )
    attention.assert_called_once()
    assert_contract(got, want)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(want)
    got.backward(upstream)
    want.backward(upstream)
    for actual, expected in zip(values, refs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)


def test_empty_and_incomplete_state_blocks():
    owner = types.SimpleNamespace(num_state_per_block=2, attn=mock.Mock())
    empty = torch.empty(2, 0, 2, 4)
    assert optimized.process_state_blocks(owner, empty, empty, empty, 0) is empty
    owner.attn.assert_not_called()
    with pytest.raises(ValueError, match="divisible"):
        optimized.process_state_blocks(owner, empty, empty, empty, 3)
    owner.attn.assert_not_called()


def test_shared_context_matches_separate_image_action_calls_and_gradients():
    torch.manual_seed(83)
    owner = types.SimpleNamespace(
        frame_seqlen=2, num_frame_per_block=2,
        num_action_per_block=2, num_state_per_block=2,
        attn=mock.Mock(side_effect=ExactAttention()),
    )
    # image QKV, action QKV, clean KV, state KV; every tensor participates
    # in training and must receive its reference gradient.
    lengths = [10, 10, 10, 4, 4, 4, 10, 10, 4, 4]
    values = [torch.randn(2, n, 2, 4, dtype=torch.float64, requires_grad=True) for n in lengths]
    refs = [value.detach().clone().requires_grad_() for value in values]
    got = optimized._process_noisy_image_action_blocks(owner, *values, 5)
    iq, ik, iv, aq, ak, av, ck, cv, sk, sv = refs
    image_parts = [ExactAttention()(iq[:, :2], ik[:, :2], iv[:, :2])]
    action_parts = []
    for block in range(2):
        image_start, action_start = 2 + block * 4, block * 2
        keys = torch.cat((ck[:, :image_start], ik[:, image_start:image_start + 4],
                          ak[:, action_start:action_start + 2], sk[:, action_start:action_start + 2]), 1)
        values_block = torch.cat((cv[:, :image_start], iv[:, image_start:image_start + 4],
                                 av[:, action_start:action_start + 2], sv[:, action_start:action_start + 2]), 1)
        image_parts.append(ExactAttention()(iq[:, image_start:image_start + 4], keys, values_block))
        action_parts.append(ExactAttention()(aq[:, action_start:action_start + 2], keys, values_block))
    want = torch.cat(image_parts, 1), torch.cat(action_parts, 1)
    assert owner.attn.call_count == 3  # conditioning + one per block
    gradients = []
    for actual, expected in zip(got, want):
        assert_contract(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        gradients.append(torch.randn_like(actual))
    sum((x * g).sum() for x, g in zip(got, gradients)).backward()
    sum((x * g).sum() for x, g in zip(want, gradients)).backward()
    for actual, expected in zip(values, refs):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)


def test_register_frequency_order_offsets_and_no_registers():
    owner = types.SimpleNamespace(num_action_per_block=2, num_state_per_block=1)
    image = torch.full((4, 1, 8), -1.)
    action, state = (torch.arange(80).reshape(10, 8) + offset for offset in (0, 100))
    assert optimized._prepare_rope_frequencies(owner, image, action, state, None) is image
    for registers, position, action_slice, state_slice in (
        (6, None, slice(0, 4), slice(0, 2)), (3, 2, slice(4, 6), slice(2, 3)),
    ):
        actual = optimized._prepare_rope_frequencies(owner, image, action, state, registers, position)
        expected = torch.cat((image, action[action_slice, None], state[state_slice, None]))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(AssertionError):
        optimized._prepare_rope_frequencies(owner, image, action, state, 6, 2)


@pytest.mark.parametrize("position", [None, 2])
def test_tensorrt_calls_original_rotation_helpers(position):
    owner = types.SimpleNamespace(num_action_per_block=2, num_state_per_block=1)
    q, k, v = (torch.randn(2, 4, 2, 8) for _ in range(3))
    freqs, action, state = object(), object(), object()
    rotate = mock.Mock(side_effect=lambda value, **kwargs: value.double())
    causal = mock.Mock(side_effect=lambda value, **kwargs: value.double())
    real = mock.Mock()
    with _fake_module(
        "groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk",
        rope_action_apply=rotate, causal_rope_action_apply=causal,
    ), mock.patch.object(
        optimized, "ENABLE_TENSORRT", True,
    ):
        got = optimized._rotate_qk(
            owner, q, k, v, freqs, action, state, 3, position, apply_rope_qk=real,
        )
    used, unused = (rotate, causal) if position is None else (causal, rotate)
    assert used.call_count == 2
    unused.assert_not_called()
    real.assert_not_called()
    kwargs = dict(
        freqs=freqs, freqs_action=action, freqs_state=state,
        action_register_length=3, num_action_per_block=2, num_state_per_block=1,
    )
    if position is not None:
        kwargs["action_state_index"] = position
    used.assert_has_calls([mock.call(q, **kwargs), mock.call(k, **kwargs)])
    for actual, expected in zip(got, (q, k)):
        assert_contract(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
