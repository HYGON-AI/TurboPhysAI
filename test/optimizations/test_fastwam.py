# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU numerical tests using tiny real upstream experts, without weights/data.

Install the FastWAM model dependencies and expose its src/ on PYTHONPATH.
No model download, simulator or accelerator is needed for these tests.
"""

import importlib
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("fastwam.models.wan22.fastwam")
pytestmark = pytest.mark.model_deps

UP = "fastwam.models.wan22"
OPT = "turbo_physai.optimizations.models.fastwam"


def module(prefix, name):
    return importlib.import_module(f"{prefix}.{name}")


def experts(prefix, checkpoint=False):
    shared = dict(hidden_dim=16, ffn_dim=32, text_dim=12, freq_dim=8,
                  eps=1e-6, num_heads=2, attn_head_dim=12, num_layers=1,
                  use_gradient_checkpointing=checkpoint)
    video = module(prefix, "wan_video_dit").WanVideoDiT(
        **shared, in_dim=4, out_dim=4, patch_size=(1, 2, 2),
        has_image_input=False, seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = module(prefix, "action_dit").ActionDiT(**shared, action_dim=3)
    return video, action


@pytest.fixture(autouse=True)
def eager(monkeypatch):
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    torch.set_num_threads(1)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("variant,cls_name", [
    ("fastwam", "FastWAM"), ("fastwam_joint", "FastWAMJoint"),
    ("fastwam_idm", "FastWAMIDM"), ("fastwam_optional_idm", "FastWAMOptionalIDM"),
])
def test_training_loss_and_all_parameter_gradients(variant, cls_name, checkpoint):
    def make(prefix):
        use_checkpoint = checkpoint and prefix == OPT
        video, action = experts(prefix, use_checkpoint)
        mot = module(prefix, "mot").MoT(
            {"video": video, "action": action}, mot_checkpoint_mixed_attn=use_checkpoint,
        )
        model = getattr(module(prefix, variant), cls_name)(
            video, action, mot, torch.nn.Identity(), text_dim=12,
        )
        model.video_cond_noise_prob = 0.5
        model.action_idm_prob = 0.5
        return model

    torch.manual_seed(7)
    reference, optimized = make(UP), make(OPT)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    latent = torch.randn(2, 4, 2, 4, 4)
    inputs = dict(
        input_latents=latent, first_frame_latents=latent[:, :, :1].clone(),
        context=torch.randn(2, 3, 12),
        context_mask=torch.tensor([[True, True, False], [True, True, True]]),
        action=torch.randn(2, 4, 3),
        action_is_pad=torch.tensor([[False, False, True, True], [False] * 4]),
        image_is_pad=None, fuse_vae_embedding_in_latents=True,
    )
    losses = []
    for model in (reference, optimized):
        model.build_inputs = lambda sample, tiled=False: inputs
        torch.manual_seed(123)
        loss, _ = model.training_loss({})
        loss.backward()
        losses.append(loss.detach())
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    ref_params = dict(reference.named_parameters())
    for name, param in optimized.named_parameters():
        expected = ref_params[name].grad
        assert (param.grad is None) == (expected is None), name
        if expected is not None:
            torch.testing.assert_close(param.grad, expected, rtol=2e-4, atol=2e-6, msg=name)


@pytest.mark.parametrize("expert_index", [0, 1])
def test_rope_preserves_complex_dtype(expert_index):
    expert = experts(OPT)[expert_index]
    before = expert.freqs
    expert.to(dtype=torch.bfloat16)
    freqs = expert.freqs if isinstance(expert.freqs, tuple) else (expert.freqs,)
    for table in freqs:
        assert table.is_complex()
        assert table.device == next(expert.parameters()).device
    assert expert.freqs is before  # dtype-only casts must not regenerate the cache


def test_video_forward_supports_checkpointing_and_optional_mask():
    model = experts(OPT, checkpoint=True)[0]
    x = torch.randn(2, 4, 2, 4, 4, requires_grad=True)
    output = model(x, torch.ones(2), torch.randn(2, 3, 12),
                   fuse_vae_embedding_in_latents=True)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_action_forward_matches_reference_with_optional_mask():
    ref, opt = experts(UP)[1], experts(OPT, checkpoint=True)[1]
    opt.load_state_dict(ref.state_dict())
    x = torch.randn(2, 4, 3)
    t = torch.ones(2)
    context = torch.randn(2, 3, 12)
    torch.testing.assert_close(ref(x, t, context), opt(x, t, context))


def test_vae_batch_loop_keeps_order_and_shape():
    from turbo_physai.optimizations.models.fastwam.vae import encode_videos

    videos = torch.randn(3, 3, 5, 4, 4)
    vae = SimpleNamespace(single_encode=lambda video, device: video[:, :2, ::4])
    actual = encode_videos(vae, videos, "cpu")
    torch.testing.assert_close(actual, videos[:, :2, ::4])
    with pytest.raises(NotImplementedError, match="Tiled"):
        encode_videos(vae, videos, "cpu", tiled=True)


def test_vae38_encoder_matches_upstream():
    from turbo_physai.optimizations.models.fastwam.vae import encode_vae38
    upstream = module(UP, "wan_video_vae").VideoVAE38_.encode

    class TinyVAE:
        z_dim = 2

        def clear_cache(self):
            self._enc_feat_map = []

        def encoder(self, x, feat_cache, feat_idx):
            return x[:, :4], feat_cache, feat_idx

        def conv1(self, x):
            return x

    x = torch.randn(2, 3, 5, 4, 4)
    scale = (torch.randn(2), torch.rand(2) + 0.1)
    torch.testing.assert_close(upstream(TinyVAE(), x, scale), encode_vae38(TinyVAE(), x, scale))


@pytest.mark.parametrize("variant,cls_name", [
    ("fastwam", "FastWAM"), ("fastwam_joint", "FastWAMJoint"),
    ("fastwam_idm", "FastWAMIDM"),
])
def test_action_inference_matches_upstream(variant, cls_name):
    def make(prefix):
        video, action = experts(prefix)
        mot = module(prefix, "mot").MoT({"video": video, "action": action}, False)
        vae = torch.nn.Module()
        vae.temporal_downsample_factor = 4
        vae.upsampling_factor = 8
        vae.model = SimpleNamespace(z_dim=4)
        model = getattr(module(prefix, variant), cls_name)(video, action, mot, vae, text_dim=12)
        model._encode_input_image_latents_tensor = lambda **kwargs: first_frame.clone()
        model._decode_latents = lambda latents, **kwargs: latents
        return model

    first_frame = torch.randn(1, 4, 1, 2, 2)
    reference, optimized = make(UP), make(OPT)
    optimized.load_state_dict(reference.state_dict(), strict=True)
    kwargs = dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                  action_horizon=4, context=torch.randn(1, 3, 12),
                  context_mask=torch.ones(1, 3, dtype=torch.bool),
                  num_inference_steps=2, seed=31)
    if variant in {"fastwam_idm", "fastwam_joint"}:
        kwargs["num_video_frames"] = 5
    actual = optimized.infer_action(**kwargs)["action"]
    expected = reference.infer_action(**kwargs)["action"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_idm_factory_keeps_configurable_probability():
    from turbo_physai.optimizations.models.fastwam.fastwam_idm import FastWAMIDM
    from turbo_physai.optimizations.models.fastwam.fastwam_joint import FastWAMJoint

    with patch.object(FastWAMJoint, "from_wan22_pretrained", return_value=SimpleNamespace()):
        model = FastWAMIDM.from_wan22_pretrained(video_cond_noise_prob=0.25)
        assert model.video_cond_noise_prob == 0.25
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        FastWAMIDM.from_wan22_pretrained(video_cond_noise_prob=2)


@pytest.mark.parametrize("variant,cls_name", [
    ("fastwam", "FastWAM"), ("fastwam_joint", "FastWAMJoint"),
    ("fastwam_idm", "FastWAMIDM"),
])
def test_legacy_factory_arguments_build_optimized_model(variant, cls_name):
    video, action = experts(OPT)
    base = module(OPT, "fastwam")
    components = SimpleNamespace(
        dit=video, vae=torch.nn.Identity(), text_encoder=None, tokenizer=None,
        dit_path="test", vae_path="test", text_encoder_path=None, tokenizer_path=None,
    )
    cls = getattr(module(OPT, variant), cls_name)
    with patch.object(base, "load_wan22_ti2v_5b_components", return_value=components), patch.object(
        base.ActionDiT, "from_pretrained", return_value=action
    ):
        kwargs = dict(device="cpu", torch_dtype=torch.float32,
                      video_dit_config={"text_dim": 12}, compile_training_denoise=False)
        if variant == "fastwam_idm":
            kwargs["video_cond_noise_prob"] = 0.3
        model = cls.from_wan22_pretrained(**kwargs)
        assert model.video_expert is video
        assert model.action_expert is action
        assert not getattr(model, "compile_training_denoise", False)
        assert isinstance(model.mot, module(OPT, "mot").MoT)


def test_packaged_config_applies_to_real_upstream_in_fresh_process():
    code = '''
        import turbo_physai
        from turbo_physai.optimizations.models.fastwam import catalog, loader, fastwam_idm
        report = turbo_physai.apply(model="fastwam")
        assert report.summary["applied"] == 2, report
        from fastwam.models.wan22.fastwam import FastWAM
        from fastwam.models.wan22.fastwam_joint import FastWAMJoint
        from fastwam.models.wan22.fastwam_idm import FastWAMIDM
        from fastwam.models.wan22.fastwam_optional_idm import FastWAMOptionalIDM
        assert issubclass(FastWAMJoint, FastWAM)
        assert issubclass(FastWAMIDM, FastWAMJoint)
        assert issubclass(FastWAMOptionalIDM, FastWAMIDM)
        assert FastWAMIDM is fastwam_idm.FastWAMIDM
        from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
        assert load_wan22_ti2v_5b_components is loader.load_wan22_ti2v_5b_components
        dit_entry = next(x for x in loader.WAN22_MODEL_REGISTRY if x["model_name"] == "wan_video_dit")
        from fastwam.models.wan22.wan_video_dit import WanVideoDiT
        assert dit_entry["model_class"] is WanVideoDiT
    '''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(os.path.abspath(p) for p in sys.path if p)
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("distributed", [False, True])
def test_cli_bootstraps_fastwam_training_ranks(tmp_path, distributed):
    if distributed:
        import socket

        try:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("torchrun rendezvous requires permission to open a local socket")
    probe = tmp_path / "probe.py"
    probe.write_text(textwrap.dedent('''
        import os
        from pathlib import Path
        from fastwam.models.wan22.fastwam import FastWAM
        from fastwam.models.wan22.helpers.loader import load_wan22_ti2v_5b_components
        from turbo_physai.optimizations.models.fastwam.fastwam import FastWAM as Optimized
        from turbo_physai.optimizations.models.fastwam import loader
        assert FastWAM is Optimized
        assert load_wan22_ti2v_5b_components is loader.load_wan22_ti2v_5b_components
        assert os.environ.get("TURBO_PHYSAI_ACCURACY_MODE") == "1"
        rank = os.environ.get("RANK", "0")
        Path(__file__).with_name("rank-" + rank).write_text("applied")
    '''))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(os.path.abspath(p) for p in sys.path if p)
    command = [sys.executable, "-m", "turbo_physai.cli", "run",
               "--model", "fastwam", "--accuracy", "--log-report", "--disable-numa"]
    if distributed:
        command += [sys.executable, "-m", "torch.distributed.run", "--standalone",
                    "--nproc_per_node=2"]
    else:
        command += [sys.executable]
    result = subprocess.run(command + [str(probe)], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2 if distributed else 1):
        assert (tmp_path / f"rank-{rank}").read_text() == "applied"
