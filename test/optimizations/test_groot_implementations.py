"""HIP Math SDPA, Qwen position operations and fused optimization."""

import contextlib
import copy
from types import SimpleNamespace

import pytest
import torch

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.optimizations.models.groot.catalog import DIT_HIP_MATH_SDPA
from turbo_physai.optimizations.models.groot.optim import create_optimizer_apex_wrapper
from turbo_physai.optimizations.models.groot.qwen_position import (
    fast_pos_embed_interpolate,
    rope_index_wrapper,
    rot_pos_emb,
)
from turbo_physai.optimizations.models.groot.replacements import (
    dit_sdpa_context_wrapper,
)


# Hip math sdpa

def test_group_targets_groot_dit_context():
    (spec,) = DIT_HIP_MATH_SDPA.specs
    assert spec.mechanism is Mechanism.WRAPPER
    assert spec.target == "gr00t.model.modules.dit._sdpa_context"


@pytest.mark.parametrize("mode", [None, "math"])
def test_hip_forces_math_temporarily_and_restores_backend_flags(monkeypatch, mode):
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    if mode is None:
        monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    else:
        monkeypatch.setenv("GR00T_DIT_SDPA_MODE", mode)
    flags = lambda: (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled(), torch.backends.cuda.mem_efficient_sdp_enabled())
    before = flags()
    wrapped = dit_sdpa_context_wrapper(contextlib.nullcontext, {})
    with wrapped():
        assert flags() == (True, False, False)
    assert flags() == before


@pytest.mark.parametrize("hip,mode", [(None, None), ("test-hip", "default")])
def test_other_platform_or_default_override_uses_original(monkeypatch, hip, mode):
    monkeypatch.setattr(torch.version, "hip", hip)
    if mode is None:
        monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    else:
        monkeypatch.setenv("GR00T_DIT_SDPA_MODE", mode)
    sentinel = object()
    wrapped = dit_sdpa_context_wrapper(lambda: sentinel, {})
    assert wrapped() is sentinel


def test_real_dit_replacement_is_applied_and_restored(monkeypatch):
    dit = pytest.importorskip("gr00t.model.modules.dit")
    monkeypatch.setattr(torch.version, "hip", "test-hip")
    monkeypatch.delenv("GR00T_DIT_SDPA_MODE", raising=False)
    original = dit._sdpa_context
    handler = default_handlers()[Mechanism.WRAPPER]
    prepared = handler.prepare(DIT_HIP_MATH_SDPA.specs[0], {})
    snapshot = handler.snapshot(prepared)
    try:
        handler.apply(prepared)
        with dit._sdpa_context():
            assert torch.backends.cuda.math_sdp_enabled()
            assert not torch.backends.cuda.flash_sdp_enabled()
    finally:
        handler.restore(snapshot)
    assert dit._sdpa_context is original


# Qwen position

@pytest.fixture
def visual():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    config = Qwen3VLVisionConfig(depth=1, hidden_size=32, intermediate_size=64,
                                  num_heads=4, out_hidden_size=32,
                                  num_position_embeddings=64, deepstack_visual_indexes=[])
    return Qwen3VLVisionModel(config)


@pytest.mark.parametrize("grid", [[[1, 4, 4]], [[2, 4, 6], [1, 2, 4]]])
def test_position_outputs_and_embedding_gradients_match(visual, grid):
    other = copy.deepcopy(visual)
    grid = torch.tensor(grid)
    expected = visual.fast_pos_embed_interpolate(grid)
    actual = fast_pos_embed_interpolate(other, grid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(other.pos_embed.weight.grad, visual.pos_embed.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(rot_pos_emb(other, grid), visual.rot_pos_emb(grid), rtol=0, atol=0)
    assert torch.equal(fast_pos_embed_interpolate(other, grid), actual)


def test_instance_caches_do_not_leak(visual):
    other = copy.deepcopy(visual)
    a, b = torch.tensor([[1, 4, 4]]), torch.tensor([[1, 2, 4]])
    expected = visual.fast_pos_embed_interpolate(a)
    fast_pos_embed_interpolate(visual, a)
    fast_pos_embed_interpolate(other, b)
    assert torch.equal(fast_pos_embed_interpolate(visual, a), expected)
    assert visual._gr00t_fast_pos_cache[0] != other._gr00t_fast_pos_cache[0]


def test_tracing_does_not_create_or_replace_caches(visual, monkeypatch):
    grid = torch.tensor([[1, 4, 4]])
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    fast_pos_embed_interpolate(visual, grid)
    rot_pos_emb(visual, grid)
    assert not hasattr(visual, "_gr00t_fast_pos_cache")
    assert not hasattr(visual, "_gr00t_rot_pos_cache")


def test_rope_wrapper_forwards_text_and_returns_multimodal_results():
    seen = []
    def original(self, input_ids=None, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        seen.append((input_ids, image_grid_thw, video_grid_thw, attention_mask))
        return input_ids + 1, input_ids.sum().reshape(1)
    fn = rope_index_wrapper(original, {})
    ids = torch.tensor([[1, 2]])
    assert torch.equal(fn(object(), ids)[0], ids + 1)
    grid = torch.tensor([[1, 2, 2]])
    result = fn(object(), ids, image_grid_thw=grid, attention_mask=torch.ones_like(ids))
    assert torch.equal(result[0], ids + 1)
    assert seen[-1][1].device.type == "cpu"


@pytest.mark.parametrize("multimodal", [False, True])
def test_rope_matches_transformers(multimodal):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2),
                             image_token_id=100, video_token_id=101, vision_start_token_id=102)
    model = SimpleNamespace(config=config)
    ids = torch.tensor([[5, 102, 100, 6]]) if multimodal else torch.tensor([[5, 6, 7, 0]])
    grid = torch.tensor([[1, 2, 2]]) if multimodal else None
    mask = torch.tensor([[1, 1, 1, 1]]) if multimodal else torch.tensor([[1, 1, 1, 0]])
    original = Qwen3VLModel.get_rope_index
    expected = original(model, ids, grid, attention_mask=mask)
    actual = rope_index_wrapper(original, {})(model, ids, grid, attention_mask=mask)
    for left, right in zip(actual, expected):
        assert torch.equal(left, right)


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="HCU is not available")
def test_position_on_device_matches_reference(visual):
    visual = visual.cuda().to(torch.bfloat16)
    other = copy.deepcopy(visual)
    grid = torch.tensor([[1, 4, 4]], device="cuda")
    expected = visual.fast_pos_embed_interpolate(grid)
    actual = fast_pos_embed_interpolate(other, grid)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.float().sum().backward()
    actual.float().sum().backward()
    torch.testing.assert_close(other.pos_embed.weight.grad, visual.pos_embed.weight.grad, rtol=0, atol=0)
    torch.testing.assert_close(rot_pos_emb(other, grid), visual.rot_pos_emb(grid), rtol=0, atol=0)


# Fused adam

def test_optimizer_override_is_temporary_and_respects_offload():
    seen = []
    def original(self):
        seen.append(self.args.optim)
        return "optimizer"
    fn = create_optimizer_apex_wrapper(original, {})
    trainer = SimpleNamespace(args=SimpleNamespace(optim="adamw_torch", deepspeed=None),
                              accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)))
    assert fn(trainer) == "optimizer"
    assert seen == ["adamw_apex_fused"]
    assert trainer.args.optim == "adamw_torch"
    trainer.accelerator.state.deepspeed_plugin = SimpleNamespace(deepspeed_config={"optimizer": {}})
    fn(trainer)
    assert seen[-1] == "adamw_torch"


def test_optimizer_restores_on_exception():
    def original(self):
        raise ValueError("failure")
    trainer = SimpleNamespace(args=SimpleNamespace(optim="adamw_torch", deepspeed=None),
                              accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)))
    with pytest.raises(ValueError):
        create_optimizer_apex_wrapper(original, {})(trainer)
    assert trainer.args.optim == "adamw_torch"


def test_transformers_optimizer_enum_is_supported():
    from transformers.training_args import OptimizerNames
    trainer = SimpleNamespace(args=SimpleNamespace(optim=OptimizerNames.ADAMW_TORCH, deepspeed=None),
                              accelerator=SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None)))
    assert create_optimizer_apex_wrapper(lambda self: self.args.optim, {})(trainer) == "adamw_apex_fused"
    assert trainer.args.optim is OptimizerNames.ADAMW_TORCH
