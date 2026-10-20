# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
import sys
import types

import pytest
import yaml

from turbo_physai.optimizations.models.dit4dit import action_compile, cosmos_forward, entry


def _install_fake_inductor(monkeypatch, *, fallback_random=True):
    config = types.SimpleNamespace()
    if fallback_random:
        config.fallback_random = False
    config.alignment_asserts = True
    inductor = types.ModuleType("torch._inductor")
    inductor.config = config
    monkeypatch.setitem(sys.modules, "torch._inductor", inductor)


def test_action_compile_keeps_rng_fallback_atomic(monkeypatch):
    _install_fake_inductor(monkeypatch)
    calls = []

    class Action:
        def compile(self, **kwargs):
            calls.append(kwargs)

    model = types.SimpleNamespace(
        action_model=types.SimpleNamespace(model=Action())
    )
    wrapped = action_compile.build_framework_wrapper(
        lambda cfg: model,
        {"backend": "inductor", "fallback_random": True},
    )

    assert wrapped(object()) is model
    assert calls == [{
        "backend": "inductor",
        "fullgraph": False,
        "dynamic": False,
        "options": {
            "fallback_random": True,
            "alignment_asserts": False,
        },
    }]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"backend": "eager"}, "backend=inductor"),
        ({"fallback_random": False}, "fallback_random=True"),
        ({"alignment_asserts": True}, "alignment_asserts=False"),
        ({"mode": "reduce-overhead"}, "mode=default"),
    ],
)
def test_action_compile_rejects_unsafe_options(monkeypatch, options, message):
    _install_fake_inductor(monkeypatch)
    with pytest.raises(ValueError, match=message):
        action_compile.build_framework_wrapper(lambda cfg: None, options)


def test_action_compile_requires_supported_inductor(monkeypatch):
    _install_fake_inductor(monkeypatch, fallback_random=False)
    with pytest.raises(RuntimeError, match="does not support fallback_random"):
        action_compile.build_framework_wrapper(lambda cfg: None, {})


def test_cosmos_segment_compile_disables_broken_alignment_assert(monkeypatch):
    _install_fake_inductor(monkeypatch)
    calls = []

    def install(transformer, *, capture_layer, segment_size, compile_kwargs):
        calls.append((transformer, capture_layer, segment_size, compile_kwargs))
        return "installed"

    transformer = object()
    wrapped = cosmos_forward.compile_segment_wrapper(
        install,
        {"segment_size": 6, "alignment_asserts": False},
    )

    assert wrapped(transformer, capture_layer=17) == "installed"
    assert calls == [(
        transformer,
        17,
        6,
        {
            "backend": "inductor",
            "fullgraph": False,
            "dynamic": False,
            "options": {"alignment_asserts": False},
        },
    )]


def test_cosmos_segment_compile_rejects_alignment_asserts(monkeypatch):
    _install_fake_inductor(monkeypatch)
    with pytest.raises(ValueError, match="alignment_asserts=False"):
        cosmos_forward.compile_segment_wrapper(
            lambda *args, **kwargs: None,
            {"alignment_asserts": True},
        )
    with pytest.raises(ValueError, match="mode=default"):
        cosmos_forward.compile_segment_wrapper(
            lambda *args, **kwargs: None,
            {"mode": "reduce-overhead"},
        )


def test_entry_sets_deepspeed_accumulation_consistently():
    plugin = types.SimpleNamespace(
        gradient_accumulation_steps=1,
        deepspeed_config={"gradient_accumulation_steps": 1},
    )
    accelerator = types.SimpleNamespace(
        gradient_accumulation_steps=1,
        state=types.SimpleNamespace(deepspeed_plugin=plugin),
    )

    entry._set_gradient_accumulation(accelerator, 4)

    assert accelerator.gradient_accumulation_steps == 4
    assert plugin.gradient_accumulation_steps == 4
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == 4


def test_entry_rejects_missing_deepspeed_or_invalid_accumulation():
    accelerator = types.SimpleNamespace(
        gradient_accumulation_steps=1,
        state=types.SimpleNamespace(deepspeed_plugin=None),
    )
    with pytest.raises(ValueError, match=">= 1"):
        entry._set_gradient_accumulation(accelerator, 0)
    with pytest.raises(RuntimeError, match="DeepSpeed"):
        entry._set_gradient_accumulation(accelerator, 1)


def test_dit4dit_recipe_targets_product_torch_210():
    recipe = Path(action_compile.__file__).with_name("configs") / "recipe.yaml"
    data = yaml.safe_load(recipe.read_text())
    assert data["compatibility"]["dependencies"]["torch"] == ">=2.10,<2.11"


def test_video_pack_rejects_mixed_future_frame_counts():
    torch = pytest.importorskip("torch")
    from turbo_physai.optimizations.models.dit4dit import video_pack

    images = [
        [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)],
        [torch.zeros(3, 2, 2)],
    ]
    with pytest.raises(ValueError, match="same number of future frames"):
        video_pack.build_cosmos_inputs(
            object(), images, ["first", "second"]
        )


def test_action_dropout_fallback_matches_eager_rng_on_hcu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HCU is not available")
    from torch._inductor import config

    if not hasattr(config, "fallback_random"):
        pytest.skip("fallback_random is not available")

    eager = torch.nn.Dropout(p=0.2).train().cuda()
    compiled = torch.compile(
        torch.nn.Dropout(p=0.2).train().cuda(),
        backend="inductor",
        fullgraph=False,
        dynamic=False,
        options={"fallback_random": True},
    )
    inputs = torch.ones(1_000_003, device="cuda")

    torch.manual_seed(20260915)
    expected = eager(inputs)
    expected_state = torch.cuda.get_rng_state()

    torch.manual_seed(20260915)
    actual = compiled(inputs)
    actual_state = torch.cuda.get_rng_state()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_state, expected_state)
