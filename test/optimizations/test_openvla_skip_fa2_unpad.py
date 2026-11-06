# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``openvla.llm.skip_fa2_unpad`` Group.

The Group installs ``make_fast_fa2_causal_mask_wrapper`` in place of
``LlamaModel._update_causal_mask``.  On a right-padded prefill, transformers
hands the 2D padding mask straight back on its FA2 branch, which pushes
``LlamaFlashAttention2._flash_attention_forward`` onto ``_get_unpad_data`` +
``flash_attn_varlen_func`` + ``pad_input``.  The replacement returns ``None``
instead, so the same call takes plain ``flash_attn_func(causal=True)``.

Three separate properties carry that behaviour, and each needs its own test
because none of them can observe the others:

* **It is installed and it fires.**  A test that only compares the two FA2
  arms would also pass with the Group disabled, so the padding case asserts
  the unpad path is really taken without the replacement and really skipped
  with it.
* **The dropped mask is equivalent.**  Right padding under a causal mask
  confines every real token to real tokens, so real-token outputs, the loss
  and every parameter gradient must match bit for bit.
* **Every other path is delegated.**  A wrapper that also returned ``None``
  while decoding, or under ``sdpa``/``eager``, would silently drop a real
  padding mask.  Those branches are pinned without a device by driving the
  wrapper with a stub ``original``.
"""

from __future__ import annotations

import textwrap
import types

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.engine.checking.context import detect_context
from turbo_physai.engine.checking.ordering import Preparation
from turbo_physai.engine.config.loader import load_optimization_config
from turbo_physai.engine.contracts import Decision, Mechanism
from turbo_physai.engine.definitions.registry import default_registry
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.optimizations.models.openvla.catalog import SKIP_FA2_UNPAD
from turbo_physai.optimizations.models.openvla.skip_fa2_unpad import (
    make_fast_fa2_causal_mask_wrapper,
)


GROUP_ID = "openvla.llm.skip_fa2_unpad"
TARGET = "transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask"
REPLACEMENT = (
    "turbo_physai.optimizations.models.openvla."
    "skip_fa2_unpad.make_fast_fa2_causal_mask_wrapper"
)

SEED = 20240607
LAYERS = 2
HEADS = 4
HEAD_DIM = 16
HIDDEN = HEADS * HEAD_DIM
VOCAB = 128


class _OriginalRecorder:
    """Stub ``LlamaModel._update_causal_mask`` that records how it was called."""

    def __init__(self):
        self.calls = []
        self.result = object()

    def __call__(
        self, model, attention_mask, input_tensor, cache_position, past_seen_tokens
    ):
        self.calls.append(
            (model, attention_mask, input_tensor, cache_position, past_seen_tokens)
        )
        return self.result


def _stub_model(implementation):
    """A stand-in for ``self``: the wrapper only reads ``config``."""

    return types.SimpleNamespace(
        config=types.SimpleNamespace(_attn_implementation=implementation)
    )


def _padded_mask(lengths, sequence_length):
    """2D right-padding mask: real tokens first, padding afterwards."""

    mask = torch.zeros(len(lengths), sequence_length, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    return mask


def _left_padded_mask(lengths, sequence_length):
    """2D left-padding mask: padding first, real tokens afterwards."""

    mask = torch.zeros(len(lengths), sequence_length, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, sequence_length - length :] = 1
    return mask


@pytest.mark.parametrize(
    "options", [None, {}, {"unused": "option"}], ids=["no-options", "empty", "present"]
)
def test_prefill_padding_mask_short_circuits_the_causal_mask(options):
    """FA2 prefill with a mask returns ``None`` and never reaches ``original``."""

    original = _OriginalRecorder()
    wrapper = make_fast_fa2_causal_mask_wrapper(original, options)

    causal_mask = wrapper(
        _stub_model("flash_attention_2"),
        _padded_mask((4, 2), 4),
        torch.zeros(2, 4, 8),
        torch.arange(4),
        0,
    )

    assert causal_mask is None
    assert original.calls == [], "the original must not materialise a mask"


@pytest.mark.parametrize(
    "implementation,past_seen_tokens,mask_present",
    [
        ("eager", 0, True),
        ("sdpa", 0, True),
        ("flash_attention_2", 4, True),
        ("flash_attention_2", 0, False),
    ],
    ids=["eager", "sdpa", "cached-decode", "no-mask"],
)
def test_every_other_path_delegates_to_the_original(
    implementation, past_seen_tokens, mask_present
):
    """Non-prefill, non-FA2 and mask-less calls stay bit-for-bit untouched."""

    original = _OriginalRecorder()
    wrapper = make_fast_fa2_causal_mask_wrapper(original, None)
    model = _stub_model(implementation)
    mask = _padded_mask((4, 2), 4) if mask_present else None
    input_tensor = torch.zeros(2, 1 if past_seen_tokens else 4, 8)
    cache_position = torch.tensor([4]) if past_seen_tokens else torch.arange(4)

    result = wrapper(model, mask, input_tensor, cache_position, past_seen_tokens)

    assert result is original.result
    (call,) = original.calls
    assert call[0] is model
    assert call[1] is mask
    assert call[2] is input_tensor
    assert call[3] is cache_position
    assert call[4] == past_seen_tokens


def test_left_padded_prefill_delegates_to_the_original():
    """Left padding must fall back: dropping the mask would expose the pad slots.

    With left padding the real tokens follow the pad slots, so plain causal
    attention would attend to the pad key/values instead of having them
    unpadded away by the varlen path -- a silent numerical regression.
    """

    original = _OriginalRecorder()
    wrapper = make_fast_fa2_causal_mask_wrapper(original, None)
    model = _stub_model("flash_attention_2")
    mask = _left_padded_mask((4, 2), 4)

    result = wrapper(model, mask, torch.zeros(2, 4, 8), torch.arange(4), 0)

    assert result is original.result
    (call,) = original.calls
    assert call[1] is mask


def test_unpadded_prefill_still_short_circuits_the_causal_mask():
    """A batch with no padding at all has no left padding, so it still skips."""

    original = _OriginalRecorder()
    wrapper = make_fast_fa2_causal_mask_wrapper(original, None)

    causal_mask = wrapper(
        _stub_model("flash_attention_2"),
        _padded_mask((4, 4), 4),
        torch.zeros(2, 4, 8),
        torch.arange(4),
        0,
    )

    assert causal_mask is None
    assert original.calls == []


def test_engine_prepares_the_group_against_the_real_llama_target(tmp_path):
    """The declaration resolves to the real target through the public entry point."""

    pytest.importorskip("transformers.models.llama.modeling_llama")

    (spec,) = SKIP_FA2_UNPAD.specs
    assert SKIP_FA2_UNPAD.group_id == GROUP_ID
    assert spec.mechanism is Mechanism.WRAPPER
    assert spec.target == TARGET
    assert spec.replacement == REPLACEMENT

    config = tmp_path / "optimization.yaml"
    config.write_text(
        textwrap.dedent(
            f"""
            schema_version: turbophysai/optimization-config/v1
            kind: OptimizationConfig
            metadata: {{id: openvla-skip-fa2-unpad, version: "1"}}
            optimization_groups:
              - id: {GROUP_ID}
            """
        ),
        encoding="utf-8",
    )

    # Preparation resolves the target and constructs the wrapper without
    # installing it.  The public `check()` entry point was removed upstream
    # (`refactor(api)!: remove redundant check and CLI commands`); `apply()`
    # would install the Group process-wide, which this test must not do.
    prepared = Preparation(default_registry, default_handlers()).prepare(
        run_id="openvla-skip-fa2-unpad",
        config=load_optimization_config(config),
        environment=detect_context(),
    )

    (group,) = prepared.groups
    assert group.group_id == GROUP_ID
    assert group.decision is Decision.APPLY
    assert group.members == (f"{GROUP_ID}.update_causal_mask",)
    assert prepared.conflicts == ()


def _flash_attention_llama(lengths=(8, 5, 3), sequence_length=8):
    """A small bf16 FA2 ``LlamaModel`` plus one right-padded token batch."""

    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    transformers = pytest.importorskip("transformers")
    if not torch.cuda.is_available():
        pytest.skip("the FA2 varlen kernel needs a real accelerator device")

    config = transformers.LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS // 2,
        max_position_embeddings=64,
        _attn_implementation="flash_attention_2",
    )
    model = (
        transformers.LlamaModel(config)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    assert model.config._attn_implementation == "flash_attention_2", (
        "the comparison would measure a different attention kernel"
    )

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    ids = torch.randint(
        VOCAB, (len(lengths), sequence_length), generator=generator, device="cuda"
    )
    mask = _padded_mask(lengths, sequence_length).to("cuda")
    return llama, model, ids, mask, mask.bool()


def _count_unpad_calls(monkeypatch, llama):
    """Count ``_get_unpad_data`` calls; the varlen path makes one per layer."""

    calls = []
    genuine = llama._get_unpad_data

    def spy(attention_mask):
        calls.append(attention_mask)
        return genuine(attention_mask)

    monkeypatch.setattr(llama, "_get_unpad_data", spy)
    return calls


@pytest.mark.hcu
@pytest.mark.model_deps
def test_padded_prefill_skips_the_unpad_path_and_stays_bit_identical(monkeypatch):
    """The Group removes the varlen call and changes no number that is trained on."""

    llama, model, ids, mask, real = _flash_attention_llama()
    unpad_calls = _count_unpad_calls(monkeypatch, llama)
    original = llama.LlamaModel._update_causal_mask
    replacement = make_fast_fa2_causal_mask_wrapper(original, None)

    def arm(wrapped):
        monkeypatch.setattr(
            llama.LlamaModel,
            "_update_causal_mask",
            replacement if wrapped else original,
        )
        model.zero_grad(set_to_none=True)
        unpad_calls.clear()

        output = model(input_ids=ids, attention_mask=mask).last_hidden_state
        calls = len(unpad_calls)
        loss = (output * real.unsqueeze(-1)).sum()
        loss.backward()

        gradients = {}
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"no gradient reached {name}"
            gradients[name] = parameter.grad.detach().clone()
        return output.detach(), loss.detach().clone(), gradients, calls

    default_output, default_loss, default_gradients, default_calls = arm(False)
    wrapped_output, wrapped_loss, wrapped_gradients, wrapped_calls = arm(True)

    # Without both halves the test could pass by running one code path twice:
    # the default arm must really enter varlen, and the Group must really skip it.
    assert default_calls > 0, "the default arm never reached the unpad path"
    assert wrapped_calls == 0, "the replacement did not skip the unpad path"

    assert torch.equal(default_output[real], wrapped_output[real])
    assert torch.equal(default_loss, wrapped_loss)
    assert set(default_gradients) == set(wrapped_gradients)
    assert any(
        float(gradient.abs().sum()) != 0.0 for gradient in default_gradients.values()
    ), "every gradient is zero, which would make the comparison vacuous"
    for name, gradient in default_gradients.items():
        assert torch.equal(gradient, wrapped_gradients[name]), (
            f"parameter gradient differs: {name}"
        )


@pytest.mark.hcu
@pytest.mark.model_deps
def test_unpadded_batch_never_enters_the_unpad_path(monkeypatch):
    """A fully real batch already skipped the mask, so the Group is a no-op."""

    llama, model, ids, mask, _ = _flash_attention_llama()
    unpad_calls = _count_unpad_calls(monkeypatch, llama)
    original = llama.LlamaModel._update_causal_mask
    replacement = make_fast_fa2_causal_mask_wrapper(original, None)
    unpadded = torch.ones_like(mask)

    def arm(wrapped):
        monkeypatch.setattr(
            llama.LlamaModel,
            "_update_causal_mask",
            replacement if wrapped else original,
        )
        unpad_calls.clear()
        with torch.no_grad():
            output = model(input_ids=ids, attention_mask=unpadded).last_hidden_state
        return output, len(unpad_calls)

    default_output, default_calls = arm(False)
    wrapped_output, wrapped_calls = arm(True)

    assert default_calls == 0
    assert wrapped_calls == 0
    assert torch.equal(default_output, wrapped_output)
