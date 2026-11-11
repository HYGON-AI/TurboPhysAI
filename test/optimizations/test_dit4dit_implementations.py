# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import sys
import types

import pytest

from turbo_physai.optimizations.models.dit4dit import (
    action_compile,
    cosmos_cache,
    cosmos_forward,
    entry,
    video_pack,
)


@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("fast_path", [False, True])
@pytest.mark.parametrize("value", [0, 1, 255])
def test_video_uint8_scale_independent_of_brightness(as_numpy, fast_path, value):
    torch = pytest.importorskip("torch")
    x = torch.full((6, 7, 3), value, dtype=torch.uint8)
    actual = video_pack.to_chw_float01(
        x.numpy() if as_numpy else x, assume_normalized_float=fast_path
    )
    expected = x.permute(2, 0, 1).float().contiguous() / 255.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("fast_path", [False, True])
def test_video_normalized_float_keeps_scale(as_numpy, fast_path):
    torch = pytest.importorskip("torch")
    x = torch.linspace(0, 1, 126).reshape(6, 7, 3)
    actual = video_pack.to_chw_float01(
        x.numpy() if as_numpy else x, assume_normalized_float=fast_path
    )
    torch.testing.assert_close(actual, x.permute(2, 0, 1), rtol=0, atol=0)


def _install_fake_inductor(monkeypatch, *, fallback_random=True):
    config = types.SimpleNamespace()
    if fallback_random:
        config.fallback_random = False
    config.alignment_asserts = True
    inductor = types.ModuleType("torch._inductor")
    inductor.config = config
    monkeypatch.setitem(sys.modules, "torch._inductor", inductor)


@pytest.mark.parametrize(
    "factory",
    (
        video_pack.compile_pack_wrapper,
        cosmos_forward.compile_segment_wrapper,
        action_compile.build_framework_wrapper,
    ),
)
def test_compile_wrappers_honor_global_disable(monkeypatch, factory):
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    original = lambda *args, **kwargs: (args, kwargs)

    assert factory(original, {}) is original


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


def test_high_sigma_sampling_skips_incomplete_strategy():
    torch = pytest.importorskip("torch")
    timesteps = torch.tensor([0.1, 0.4, 0.9], dtype=torch.float32)

    result = cosmos_forward._apply_high_sigma_strategy(
        timesteps,
        high_sigma_ratio=0.05,
        high_sigma_min=None,
    )

    assert result is timesteps


def test_high_sigma_sampling_applies_complete_strategy():
    torch = pytest.importorskip("torch")
    timesteps = torch.zeros(8, dtype=torch.float32)

    result = cosmos_forward._apply_high_sigma_strategy(
        timesteps,
        high_sigma_ratio=1.0,
        high_sigma_min=0.98,
    )

    assert torch.all(result >= 0.98)
    assert torch.all(result <= 1.0)


def test_hidden_capture_accepts_upstream_prepare_latents_signature():
    torch = pytest.importorskip("torch")
    captured = {}

    class PrepareLatentsCalled(Exception):
        pass

    class Owner:
        device = torch.device("cpu")
        transformer = types.SimpleNamespace(
            dtype=torch.float32,
            config=types.SimpleNamespace(in_channels=17),
        )
        vae = types.SimpleNamespace(dtype=torch.float32)
        vae_scale_factor_temporal = 4
        max_sequence_length = 8

        def encode_prompt(self, **kwargs):
            return torch.zeros((1, 1, 1)), None

        def _coerce_videos_to_bcthw(self, videos, *, height, width):
            return videos

        def prepare_latents(
            self,
            *,
            video,
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames_in,
            num_frames_out,
            dtype,
            device,
            generator=None,
            latents=None,
        ):
            captured["video"] = video
            raise PrepareLatentsCalled

    videos = torch.zeros((1, 3, 1, 16, 16))
    with pytest.raises(PrepareLatentsCalled):
        cosmos_forward.feature_extractor_forward(
            Owner(),
            prompts=["test"],
            videos=videos,
            height=16,
            width=16,
        )

    assert captured["video"] is videos


def test_prepare_latents_bcthw():
    torch = pytest.importorskip("torch")
    owner = types.SimpleNamespace(
        vae=types.SimpleNamespace(dtype=torch.float32),
        vae_scale_factor_temporal=1, vae_scale_factor_spatial=1,
    )
    with pytest.raises(ValueError, match="B,3"):
        cosmos_forward.prepare_latents(
            owner, video=torch.zeros(1, 4, 3, 2, 2), batch_size=1,
            num_channels_latents=16, height=2, width=2, num_frames_in=4,
            num_frames_out=4, dtype=torch.float32, device=torch.device("cpu"),
        )


def test_latent_stats_remain_fp32_when_module_is_cast_to_bfloat16():
    torch = pytest.importorskip("torch")

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.latents_mean = torch.tensor([1.001], dtype=torch.float32)
            self.latents_std = torch.tensor([0.499], dtype=torch.float32)

    class Owner(cosmos_cache._Caches, Base):
        pass

    owner = Owner()
    expected_mean = owner.latents_mean.clone()
    expected_std = owner.latents_std.clone()

    assert "latents_mean" not in owner._buffers
    assert "latents_std" not in owner._buffers

    owner.bfloat16()

    assert owner.latents_mean.dtype == torch.float32
    assert owner.latents_std.dtype == torch.float32
    torch.testing.assert_close(owner.latents_mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(owner.latents_std, expected_std, rtol=0, atol=0)

    mean, std = owner._get_latent_stats(
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert mean.dtype == torch.bfloat16
    assert std.dtype == torch.bfloat16
    cached_mean, cached_std = owner._get_latent_stats(
        device=torch.device("cpu"), dtype=torch.bfloat16
    )
    assert cached_mean is mean
    assert cached_std is std


def test_padding_mask_cache_is_identity_safe_and_versioned():
    torch = pytest.importorskip("torch")
    transformer = types.SimpleNamespace(_dit4dit_padding_mask_cache={})
    padding_mask = torch.zeros((1, 2, 2))

    first = cosmos_forward._get_expanded_padding_mask(
        transformer,
        padding_mask,
        spatial_size=(4, 4),
        batch_size=2,
        num_frames=3,
    )
    second = cosmos_forward._get_expanded_padding_mask(
        transformer,
        padding_mask,
        spatial_size=(4, 4),
        batch_size=2,
        num_frames=3,
    )

    assert second is first
    assert next(iter(transformer._dit4dit_padding_mask_cache.values()))[0] is padding_mask

    padding_mask.fill_(1)
    after_mutation = cosmos_forward._get_expanded_padding_mask(
        transformer,
        padding_mask,
        spatial_size=(4, 4),
        batch_size=2,
        num_frames=3,
    )

    assert after_mutation is not first
    assert torch.equal(after_mutation, torch.ones_like(after_mutation))
    assert len(transformer._dit4dit_padding_mask_cache) == 1


def test_forward_static_cache_is_identity_safe_and_versioned():
    torch = pytest.importorskip("torch")

    class CacheOwner(cosmos_cache._Caches):
        pass

    owner = CacheOwner()
    mask = torch.zeros((1, 1, 2, 2, 2))
    indicator = torch.zeros((1, 1, 2, 1, 1))
    kwargs = {
        "condition_frames": 1,
        "transformer_dtype": torch.float32,
        "conditional_frame_timestep": 0.1,
        "height": 2,
        "width": 2,
    }

    first = owner._get_forward_static_tensors(
        cond_mask=mask, cond_indicator=indicator, **kwargs
    )
    repeated = owner._get_forward_static_tensors(
        cond_mask=mask, cond_indicator=indicator, **kwargs
    )
    assert repeated is first

    different_mask = torch.ones_like(mask)
    different_indicator = torch.ones_like(indicator)
    different = owner._get_forward_static_tensors(
        cond_mask=different_mask,
        cond_indicator=different_indicator,
        **kwargs,
    )
    assert different is not first
    assert torch.equal(different[1], different_mask)
    assert torch.equal(different[2], torch.zeros_like(different_mask))
    assert torch.equal(different[3], torch.zeros_like(different_indicator))

    different_mask.zero_()
    after_mutation = owner._get_forward_static_tensors(
        cond_mask=different_mask,
        cond_indicator=different_indicator,
        **kwargs,
    )
    assert after_mutation is not different
    assert torch.equal(after_mutation[1], different_mask)
    matching_entries = [
        entry
        for entry in owner._forward_static_cache.values()
        if entry[0] is different_mask
    ]
    assert len(matching_entries) == 1
    cached_mask, cached_indicator, _ = matching_entries[0]
    assert cached_mask is different_mask
    assert cached_indicator is different_indicator


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


def test_video_pack_compile_wrapper_preserves_supported_gt_ranges(monkeypatch):
    torch = pytest.importorskip("torch")
    from turbo_physai.optimizations.models.dit4dit import video_pack

    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    compiled_pack = video_pack.compile_pack_wrapper(None, {})
    videos = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 1, 2, 2)

    for gt in (
        torch.linspace(0.0, 1.0, 12).reshape(1, 3, 1, 2, 2),
        torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 1, 2, 2),
    ):
        expected = video_pack.pack_flow_matching_video(videos, gt)
        actual = compiled_pack(videos, gt)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
