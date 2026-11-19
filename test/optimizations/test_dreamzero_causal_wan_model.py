# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""DiT training: channels-last, checkpoint removal, conditioning and gradients.

Synthetic FP64 CPU model; output/input/parameter tolerances 1e-10. Timesteps,
IDs and positional tables are metadata and do not require gradients.
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
import torch.utils.checkpoint

from turbo_physai.optimizations.models.dreamzero import (
    CausalWanModel as optimized,
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


def complex_params(length, dim, theta=10000):
    angles = torch.arange(length, dtype=torch.float64)[:, None] * (
        theta ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    )
    return torch.polar(torch.ones_like(angles), angles)


def sinusoidal_embedding(dim, positions):
    angles = positions.double()[:, None] * 10000 ** (
        -torch.arange(dim // 2, dtype=torch.float64, device=positions.device) / (dim // 2)
    )
    return torch.cat((angles.cos(), angles.sin()), -1)


class RegisterEncoder(nn.Linear):
    def forward(self, x, *metadata):
        return super().forward(x)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(8, 2)

    def forward(self, x, e):
        return self.head(x + e.squeeze(2))


class TrainingBlock(nn.Linear):
    def forward(self, x, *, e, context, **metadata):
        self.metadata = metadata
        # Mix tokens so clean video and registers affect the noisy predictions.
        return super().forward(torch.tanh(x + x.mean(1, keepdim=True) + e.mean(2) + context.mean(1, keepdim=True))), None


class TinyModel(nn.Module):
    def __init__(self, concat=False):
        super().__init__()
        self.dim, self.freq_dim, self.text_len = 8, 4, 3
        self.model_type = "i2v" if concat else "ti2v"
        self.concat_first_frame_latent = concat
        self.gradient_checkpointing = True
        self.patch_embedding = nn.Conv3d(4 if concat else 2, 8, 1)
        self.time_embedding = nn.Sequential(nn.Linear(4, 8), nn.SiLU())
        self.time_projection = nn.Linear(8, 48)
        self.text_embedding = nn.Linear(6, 8)
        self.img_emb = nn.Linear(5, 8)
        self.action_encoder = RegisterEncoder(3, 8)
        self.state_encoder = RegisterEncoder(4, 8)
        self.action_decoder = RegisterEncoder(8, 3)
        self.blocks = nn.ModuleList([TrainingBlock(8, 8), TrainingBlock(8, 8)])
        self.head = Head()
        self.freqs_action, self.freqs_state = torch.zeros(8, 4), torch.zeros(8, 4)

    def _create_freqs(self, grid_size, start_frame):
        return torch.zeros(int(grid_size.prod()), 1, 4)

    def unpatchify(self, x, grid_size):
        return x.transpose(1, 2).reshape(x.shape[0], 2, *grid_size.tolist())


def reference_train(model, inputs):
    """Baseline token/embedding equations, independent of the replacement."""
    x, clean, y = inputs["x"], inputs.get("clean_x"), inputs.get("y")

    def tokenize(video):
        if model.concat_first_frame_latent:
            video = torch.cat((video, y), 1)
        return model.patch_embedding(video).flatten(2).transpose(1, 2)

    tokens = tokenize(x)
    batch, video_length = tokens.shape[:2]
    t_video = inputs["timestep"].repeat_interleave(video_length // inputs["timestep"].shape[1], dim=1)
    times = t_video
    if "action" in inputs:
        ids = torch.zeros(batch, dtype=torch.long)
        action_tokens = model.action_encoder(inputs["action"], inputs["timestep_action"], ids)
        state_tokens = model.state_encoder(inputs["state"], ids)
        tokens = torch.cat((tokens, action_tokens, state_tokens), 1)
        t_action = inputs["timestep_action"]
        times = torch.cat((times, t_action, t_action[:, ::t_action.shape[1] // state_tokens.shape[1]]), 1)
    e = model.time_embedding(sinusoidal_embedding(4, times.flatten())).reshape(batch, -1, 8)
    modulation = model.time_projection(e).reshape(batch, -1, 6, 8)
    context = model.text_embedding(inputs["context"])
    if "clip_feature" in inputs:
        context = torch.cat((model.img_emb(inputs["clip_feature"]), context), 1)
    if clean is not None:
        tokens = torch.cat((tokenize(clean), tokens), 1)
        aug = inputs.get("aug_t", torch.zeros_like(t_video))
        clean_e = model.time_embedding(sinusoidal_embedding(4, aug.flatten())).reshape(batch, -1, 8)
        modulation = torch.cat((model.time_projection(clean_e).reshape(batch, -1, 6, 8), modulation), 1)
    for block in model.blocks:
        # Baseline block formula; no call to optimized.forward_train/helpers.
        tokens = nn.functional.linear(
            torch.tanh(tokens + tokens.mean(1, keepdim=True) + modulation.mean(2) + context.mean(1, keepdim=True)),
            block.weight, block.bias,
        )
    if clean is not None:
        tokens = tokens[:, video_length:]
    action = None
    if "action" in inputs:
        action = model.action_decoder(tokens[:, video_length:video_length + action_tokens.shape[1]], ids)
    video = model.head(tokens[:, :video_length], e[:, :video_length].unsqueeze(2))
    return model.unpatchify(video, torch.tensor(x.shape[2:])), action


@pytest.mark.parametrize(
    "teacher_forcing,registers,concat,augmented",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, True),
        (True, True, True, False),
    ],
    ids=["none", "tf", "tf-registers-augmented", "tf-registers-concat"],
)
def test_conditioning_outputs_all_gradients_layout_and_no_recompute(
    teacher_forcing, registers, concat, augmented
):
    torch.manual_seed(43)
    ref = TinyModel(concat).double()
    opt = copy.deepcopy(ref)
    inputs = dict(
        x=torch.randn(2, 2, 3, 2, 2, dtype=torch.float64),
        timestep=torch.randn(2, 3, dtype=torch.float64), timestep_action=None,
        context=torch.randn(2, 3, 6, dtype=torch.float64), seq_len=12,
    )
    differentiable = ["x", "context"]
    if teacher_forcing:
        inputs["clean_x"] = torch.randn_like(inputs["x"])
        differentiable.append("clean_x")
    if augmented:
        inputs["aug_t"] = torch.randn(2, 12, dtype=torch.float64)
    if registers:
        inputs.update(
            action=torch.randn(2, 4, 3, dtype=torch.float64),
            state=torch.randn(2, 2, 4, dtype=torch.float64),
            timestep_action=torch.randn(2, 4, dtype=torch.float64),
            embodiment_id=torch.tensor([1, 2]),
        )
        differentiable += ["action", "state"]
    if concat:
        inputs.update(y=torch.randn_like(inputs["x"]), clip_feature=torch.randn(2, 2, 5, dtype=torch.float64))
        differentiable += ["y", "clip_feature"]
    ref_inputs = {
        key: value.detach().clone().requires_grad_(key in differentiable) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    opt_inputs = {
        key: value.detach().clone().requires_grad_(key in differentiable) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }
    expected = reference_train(ref, ref_inputs)
    layouts, calls = [], []
    hook = opt.patch_embedding.register_forward_pre_hook(
        lambda module, args: layouts.append(args[0].is_contiguous(memory_format=torch.channels_last_3d))
    )
    handles = [block.register_forward_hook(lambda *args: calls.append(1)) for block in opt.blocks]
    try:
        with _fake_module(
            "groot.vla.model.dreamzero.modules.wan2_1_submodule", sinusoidal_embedding_1d=sinusoidal_embedding,
        ), mock.patch.object(
            torch.utils.checkpoint, "checkpoint", side_effect=AssertionError("checkpoint must be removed"),
        ):
            actual = optimized.forward_train(opt, **opt_inputs)
            assert isinstance(actual, tuple)
            assert len(actual) == 2
            for got, want in zip(actual, expected):
                if want is None:
                    assert got is None
                else:
                    assert_contract(got, want)
                    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-10)
            upstream = [torch.randn_like(value) if value is not None else None for value in actual]
            sum((a * g).sum() for a, g in zip(actual, upstream) if a is not None).backward()
        sum((a * g).sum() for a, g in zip(expected, upstream) if a is not None).backward()
        assert all(layouts)
        assert len(layouts) == (2 if teacher_forcing else 1)
        assert len(calls) == len(opt.blocks)
        assert opt.blocks[0].metadata["is_tf"] == teacher_forcing
        assert opt.blocks[0].metadata["action_register_length"] == (6 if registers else None)
        for key in differentiable:
            assert opt_inputs[key].grad is not None, key
            torch.testing.assert_close(opt_inputs[key].grad, ref_inputs[key].grad, rtol=1e-10, atol=1e-10)
        assert_parameter_grads(opt, ref, rtol=1e-10, atol=1e-10)
    finally:
        hook.remove()
        for handle in handles:
            handle.remove()


def test_weight_initialization_preserves_values_and_changes_patch_layout():
    torch.manual_seed(71)
    ref = TinyModel().double()
    ref.patch_embedding = nn.Conv3d(2, 8, 3, padding=1).double()
    assert not ref.patch_embedding.weight.is_contiguous(memory_format=torch.channels_last_3d)
    opt = copy.deepcopy(ref)
    torch.manual_seed(13)
    for module in ref.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    nn.init.xavier_uniform_(ref.patch_embedding.weight.flatten(1))
    for embedding in (ref.text_embedding, ref.time_embedding):
        for module in embedding.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
    nn.init.zeros_(ref.head.head.weight)
    torch.manual_seed(13)
    optimized.init_weights(opt)
    for key, value in ref.state_dict().items():
        torch.testing.assert_close(opt.state_dict()[key], value, rtol=0, atol=0)
    assert opt.patch_embedding.weight.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize("enable_tensorrt", ["False", "tRuE"])
def test_rope_init_converts_tables_and_tensorrt_keeps_original(enable_tensorrt):
    def original(owner, dim, *, heads):
        owner.dim, owner.num_heads = dim, heads
        owner.freqs, owner.freqs_action, owner.freqs_state = object(), object(), object()
        owner.original_tables = (owner.freqs, owner.freqs_action, owner.freqs_state)

    original_mock = mock.Mock(side_effect=original)
    owner = types.SimpleNamespace()
    with mock.patch.dict("os.environ", {"ENABLE_TENSORRT": enable_tensorrt}):
        wrapped = optimized.compile_real_rope_freqs(original_mock, {})
        assert wrapped(owner, 48, heads=2) is None
    original_mock.assert_called_once_with(owner, 48, heads=2)
    assert wrapped.__wrapped__ is original_mock
    if enable_tensorrt.lower() == "true":
        for actual, expected in zip(
            (owner.freqs, owner.freqs_action, owner.freqs_state), owner.original_tables
        ):
            assert actual is expected
    else:
        for table, length, dim in zip(
            [owner.freqs_action, owner.freqs_state, *owner.freqs],
            [10240, 1024, 1024, 1024, 1024], [24, 24, 8, 8, 8],
        ):
            expected = torch.view_as_real(complex_params(length, dim)).flatten(-2)
            assert_contract(table, expected)
            # Up to 10240 positions amplify equivalent power/reciprocal
            # FP64 rounding; 2e-12 absolute tolerance covers this.
            torch.testing.assert_close(table, expected, rtol=1e-12, atol=2e-12)
