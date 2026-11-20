"""Compilation wrappers, Qwen text compilation and KV cache controls."""

import copy
from types import SimpleNamespace

import pytest
import torch

from turbo_physai.optimizations.models.groot.attention import (
    flash_attn_dynamo_disable_wrapper,
)
from turbo_physai.optimizations.models.groot.compile import compile_method_wrapper
from turbo_physai.optimizations.models.groot.compile_text import compile_text_wrapper
from turbo_physai.optimizations.models.groot.kv_cache import disable_kv_cache_wrapper


# Compile wrappers

def test_compile_mode_and_disable(monkeypatch):
    calls = []
    def compile_fn(fn, **kwargs):
        calls.append(kwargs)
        return fn
    monkeypatch.setattr(torch, "compile", compile_fn)
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    def f(x): return x + 1
    assert compile_method_wrapper(f, {"mode": "default"})(2) == 3
    assert calls == [{"mode": "default"}]
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert compile_method_wrapper(f, {}) is f


def test_attention_wrapper_is_disabled_for_dynamo():
    def f(x): return x + 1
    wrapped = flash_attn_dynamo_disable_wrapper(f, {})
    assert wrapped(2) == 3
    assert getattr(wrapped, "_torchdynamo_disable", False)


def test_dynamo_eager_preserves_output_and_gradients(monkeypatch):
    compiler = torch.compile
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: compiler(fn, backend="eager"))
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    def original(value, weight):
        return (value @ weight).sin()
    wrapped = compile_method_wrapper(original, {})
    x = torch.randn(3, 4, requires_grad=True)
    w = torch.randn(4, 5, requires_grad=True)
    expected = original(x, w)
    grads = torch.autograd.grad(expected.sum(), (x, w))
    actual = wrapped(x, w)
    actual_grads = torch.autograd.grad(actual.sum(), (x, w))
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(actual_grads, grads):
        torch.testing.assert_close(actual_grad, expected_grad)


# Compile qwen text layer

def test_text_compile_preserves_outputs_and_gradients(monkeypatch):
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextDecoderLayer

    cfg = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                            head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    baseline = Qwen3VLTextModel(cfg).train()
    actual = copy.deepcopy(baseline)
    inputs = torch.tensor([[1, 2, 3]])
    expected = baseline(input_ids=inputs, use_cache=False, output_hidden_states=True)
    expected.last_hidden_state.square().sum().backward()
    original_compile = torch.compile
    monkeypatch.setattr(torch, "compile", lambda f, **kwargs: original_compile(f, backend="eager"))
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compile_method_wrapper(Qwen3VLTextDecoderLayer.forward, {}))
    result = actual(input_ids=inputs, use_cache=False, output_hidden_states=True)
    torch.testing.assert_close(result.last_hidden_state, expected.last_hidden_state)
    assert len(result.hidden_states) == len(expected.hidden_states)
    for result_hidden, expected_hidden in zip(result.hidden_states, expected.hidden_states):
        torch.testing.assert_close(result_hidden, expected_hidden)
    result.last_hidden_state.square().sum().backward()
    for a, b in zip(actual.parameters(), baseline.parameters()):
        if a.grad is None or b.grad is None:
            assert a.grad is None and b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad)


def test_text_compile_dependencies():
    from turbo_physai.optimizations.models.groot.catalog import COMPILE_TEXT
    assert set(COMPILE_TEXT.definition.depends_on) == {
        "groot.qwen.flash_attn_dynamo_disable", "groot.qwen.disable_kv_cache"
    }


def test_text_compile_records_layer_calls_and_respects_disable(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    def forward(self, x): return x + 1
    wrapped = compile_text_wrapper(forward, {})
    layer = SimpleNamespace(layer_idx=3)
    assert wrapped(layer, 1) == 2
    assert wrapped(layer, 2) == 3
    assert layer._turbo_physai_text_compile_calls == 2
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert compile_text_wrapper(forward, {}) is forward


def test_sixteen_text_layers_without_cache_share_compiled_graph(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    model = Qwen3VLTextModel(config).eval()
    graph_count = []
    def backend(graph, inputs):
        graph_count.append(graph)
        return graph.forward
    torch._dynamo.reset()
    compiled = torch.compile(Qwen3VLTextDecoderLayer.forward, backend=backend)
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compiled)
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False)
    assert len(graph_count) == 1


def test_sixteen_text_layers_with_cache_hit_recompile_limit(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    model = Qwen3VLTextModel(config).eval()
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", torch.compile(Qwen3VLTextDecoderLayer.forward, backend=backend))
    with torch._dynamo.config.patch(recompile_limit=8, fail_on_recompile_limit_hit=True), torch.no_grad():
        with pytest.raises(torch._dynamo.exc.FailOnRecompileLimitHit, match="recompile_limit"):
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    assert len(graphs) == 8


def test_text_compile_with_gradient_checkpointing(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel
    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    model = Qwen3VLTextModel(config).train()
    model.gradient_checkpointing_enable()
    compiler = torch.compile
    monkeypatch.setattr(torch, "compile", lambda f, **kwargs: compiler(f, backend="eager"))
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compile_method_wrapper(Qwen3VLTextDecoderLayer.forward, {}))
    output = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False, output_hidden_states=True)
    output.last_hidden_state.square().sum().backward()
    assert len(output.hidden_states) == 3
    assert model.embed_tokens.weight.grad is not None


@pytest.mark.hcu
@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="requires an HCU")
def test_text_inductor_on_hcu_matches_eager_outputs_and_gradients(monkeypatch):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLTextModel

    config = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=16, num_attention_heads=2, num_key_value_heads=2,
                               head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]})
    expected = Qwen3VLTextModel(config).to("cuda").train()
    actual = copy.deepcopy(expected)
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    output = expected(input_ids=tokens, use_cache=False, output_hidden_states=True)
    output.last_hidden_state.square().sum().backward()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    monkeypatch.delenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", raising=False)
    monkeypatch.setattr(Qwen3VLTextDecoderLayer, "forward", compile_text_wrapper(Qwen3VLTextDecoderLayer.forward, {}))
    compiled = actual(input_ids=tokens, use_cache=False, output_hidden_states=True)
    compiled.last_hidden_state.square().sum().backward()
    for actual_hidden, expected_hidden in zip(compiled.hidden_states, output.hidden_states):
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=1e-3, atol=1e-3)
    for actual_param, expected_param in zip(actual.parameters(), expected.parameters()):
        if actual_param.grad is None or expected_param.grad is None:
            assert actual_param.grad is None and expected_param.grad is None
        else:
            torch.testing.assert_close(actual_param.grad, expected_param.grad, rtol=1e-3, atol=1e-3)
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] == 1


# Kv cache

@pytest.fixture
def text_model():
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    config = Qwen3VLTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    return Qwen3VLTextModel(config).train()


def test_default_cache_disabled_without_changing_hidden_states(text_model):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    tokens = torch.tensor([[1, 2, 3]])
    original = Qwen3VLTextModel.forward
    baseline = original(text_model, input_ids=tokens, output_hidden_states=True)
    assert text_model.config.use_cache is True
    assert baseline.past_key_values is not None
    output = disable_kv_cache_wrapper(original, {})(text_model, input_ids=tokens, output_hidden_states=True)
    assert output.past_key_values is None
    assert torch.equal(output.last_hidden_state, baseline.last_hidden_state)
    assert all(torch.equal(a, b) for a, b in zip(output.hidden_states, baseline.hidden_states))


def test_explicit_cache_and_existing_cache_are_not_overridden(text_model):
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    fn = disable_kv_cache_wrapper(Qwen3VLTextModel.forward, {})
    tokens = torch.tensor([[1, 2]])
    assert fn(text_model, input_ids=tokens, use_cache=True).past_key_values is not None
    cache = DynamicCache(config=text_model.config)
    assert fn(text_model, input_ids=tokens, past_key_values=cache).past_key_values is cache


def test_positional_explicit_use_cache_is_respected():
    def original(self, input_ids=None, attention_mask=None, past_key_values=None, use_cache=None):
        return use_cache, past_key_values
    wrapped = disable_kv_cache_wrapper(original, {})
    assert wrapped(SimpleNamespace()) == (False, None)
    assert wrapped(SimpleNamespace(), None, None, None, True) == (True, None)
    assert wrapped(SimpleNamespace(), use_cache=None) == (None, None)
