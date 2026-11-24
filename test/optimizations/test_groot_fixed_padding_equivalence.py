"""Numerical equivalence of fixed sequence padding in GR00T.

The baseline action head calls ``vl_self_attention(backbone_features)`` without an attention mask,
while ``AlternateVLDiT`` receives ``encoder_attention_mask`` and ignores masked positions. These
tests use tiny random modules to measure both stages separately.
"""

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
dit_module = pytest.importorskip("gr00t.model.modules.dit")

HEADS, HEAD_DIM, REAL, EXTRA, ACTION_LEN = 4, 8, 10, 6, 5
DIM = HEADS * HEAD_DIM


@pytest.fixture
def modules():
    torch.manual_seed(0)
    vl = dit_module.SelfAttentionTransformer(
        num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, num_layers=2, dropout=0.0,
        output_dim=DIM, positional_embeddings=None).eval()
    dit = dit_module.AlternateVLDiT(
        num_attention_heads=HEADS, attention_head_dim=HEAD_DIM, output_dim=16, num_layers=4,
        dropout=0.0, cross_attention_dim=DIM, interleave_self_attention=True,
        positional_embeddings=None, num_embeds_ada_norm=1000).eval()
    return vl, dit


@pytest.fixture
def batch():
    torch.manual_seed(1)
    batch_size = 2
    features = torch.randn(batch_size, REAL, DIM)
    image_mask = torch.zeros(batch_size, REAL, dtype=torch.bool)
    image_mask[:, 2:7] = True
    attention = torch.ones(batch_size, REAL, dtype=torch.bool)
    # The values the backbone produces at pad positions are arbitrary but not zero.
    pad_features = torch.randn(batch_size, EXTRA, DIM)
    return {
        "features": features, "image_mask": image_mask, "attention": attention,
        "padded_features": torch.cat([pad_features, features], dim=1),
        "padded_image_mask": torch.cat([torch.zeros_like(image_mask[:, :EXTRA]), image_mask], 1),
        "padded_attention": torch.cat([torch.zeros_like(attention[:, :EXTRA]), attention], 1),
        "action": torch.randn(batch_size, ACTION_LEN, DIM),
        "timestep": torch.tensor([3, 7]),
    }


def run_dit(dit, batch, *, padded):
    prefix = "padded_" if padded else ""
    features = batch[f"{prefix}features"]
    output, _ = dit(
        hidden_states=batch["action"], encoder_hidden_states=features,
        timestep=batch["timestep"], encoder_attention_mask=batch[f"{prefix}attention"],
        return_all_hidden_states=True, image_mask=batch[f"{prefix}image_mask"],
        backbone_attention_mask=batch[f"{prefix}attention"])
    return output


def parameter_gradients(module, loss):
    module.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}


def test_dit_cross_attention_ignores_left_padded_positions(modules, batch):
    _, dit = modules
    plain = run_dit(dit, batch, padded=False)
    padded = run_dit(dit, batch, padded=True)
    torch.testing.assert_close(padded, plain, rtol=1e-4, atol=1e-5)
    plain_grads = parameter_gradients(dit, plain.square().mean())
    padded_grads = parameter_gradients(dit, padded.square().mean())
    assert plain_grads.keys() == padded_grads.keys() and plain_grads
    for name, grad in plain_grads.items():
        torch.testing.assert_close(padded_grads[name], grad, rtol=1e-4, atol=1e-5, msg=name)


def test_vl_self_attention_output_depends_on_padding(modules, batch):
    """Characterization: ``vl_self_attention`` has no mask, so pad positions leak into real ones."""
    vl, _ = modules
    plain = vl(batch["features"])
    padded = vl(batch["padded_features"])[:, EXTRA:]
    relative = (padded - plain).norm() / plain.norm()
    assert relative > 1e-3, f"expected padding to change the output, relative difference {relative:.2e}"


@pytest.mark.model_deps
@pytest.mark.parametrize("compile_vl", [False, True])
def test_pad_156_preserves_native_width_vl_numerics(monkeypatch, compile_vl):
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    from turbo_physai.engine.checking.conflicts import find_group_conflicts
    from turbo_physai.engine.definitions.registry import default_registry
    from turbo_physai.engine.execution.replacements import default_handlers
    from turbo_physai.optimizations.models.groot import catalog

    declarations = [catalog.FIXED_PADDING]
    if compile_vl:
        declarations.append(catalog.COMPILE_VL)
    enabled_ids = [group.group_id for group in declarations]
    assert not find_group_conflicts(default_registry, enabled_ids)

    torch.manual_seed(3)
    kwargs = dict(num_attention_heads=2, attention_head_dim=8, num_layers=2,
                  dropout=0.2, final_dropout=True, positional_embeddings=None)
    baseline = dit_module.SelfAttentionTransformer(**kwargs).train()
    plain_forward = baseline.forward
    features = torch.randn(2, 156, 16)
    columns = torch.arange(156)
    dit = dit_module.AlternateVLDiT(
        num_attention_heads=2, attention_head_dim=8, num_layers=4, output_dim=8,
        dropout=0.2, final_dropout=True, cross_attention_dim=16,
        positional_embeddings=None, interleave_self_attention=True,
    ).train()
    action = torch.randn(2, ACTION_LEN, 16)
    timestep = torch.tensor([3, 7])
    image = ((columns >= 10) & (columns < 74)).expand(2, -1)

    def run(vl, func, valid, image_mask, width):
        torch.manual_seed(123)
        features = func()
        out, _ = dit(
            hidden_states=action, encoder_hidden_states=features, timestep=timestep,
            encoder_attention_mask=valid, backbone_attention_mask=valid, image_mask=image_mask,
            return_all_hidden_states=True,
        )
        dit.zero_grad(set_to_none=True)
        loss = out.square().mean()
        for name, tensor in (("VL output", features), ("DiT output", out), ("loss", loss)):
            assert torch.isfinite(tensor).all(), f"{name} has NaN or Inf at width {width}"
        vl_grads = parameter_gradients(vl, loss)
        dit_grads = {name: p.grad.clone() for name, p in dit.named_parameters() if p.grad is not None}
        for module, grads in (("VL", vl_grads), ("DiT", dit_grads)):
            for name, grad in grads.items():
                assert torch.isfinite(grad).all(), f"{module}.{name} gradient has NaN or Inf at width {width}"
        return features[:, -width:], out, loss, vl_grads, dit_grads, torch.get_rng_state()

    expected = {}
    for width in (152, 149, 156):
        valid = columns >= torch.tensor([[156 - width], [159 - width]])
        expected[width] = run(
            baseline, lambda: baseline(features[:, -width:]), valid[:, -width:],
            image[:, -width:], width,
        )

    graphs = []
    if compile_vl:
        compile_original = torch.compile

        def counting_backend(graph, inputs, **_):
            graphs.append(graph)
            return graph.forward

        def compile_for_test(fn, **kwargs):
            return compile_original(fn, backend=counting_backend, **kwargs)

        monkeypatch.setattr(torch, "compile", compile_for_test)

    def trainer_step(self, model, inputs):
        return model(**self._prepare_inputs(inputs))

    monkeypatch.setattr(Gr00tTrainer, "training_step", trainer_step)
    trainer = object.__new__(Gr00tTrainer)
    trainer.args = SimpleNamespace(device=torch.device("cpu"), past_index=-1)
    trainer.is_deepspeed_enabled = False
    trainer.data_collator = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    )
    handlers = default_handlers()
    with ExitStack() as cleanup:
        for declaration in declarations:
            for spec in declaration.specs:
                handler = handlers[spec.mechanism]
                item = handler.prepare(spec, {})
                cleanup.callback(handler.restore, handler.snapshot(item))
                handler.apply(item)
        vl = dit_module.SelfAttentionTransformer(**kwargs).train()
        vl.load_state_dict(baseline.state_dict())
        head = SimpleNamespace(vlln=torch.nn.Identity(), vl_self_attention=vl)

        def collated_batch(width):
            valid = columns >= torch.tensor([[156 - width], [159 - width]])
            return {"inputs": {
                "input_ids": torch.ones(2, width, dtype=torch.long),
                "attention_mask": valid[:, -width:].long(),
            }}

        def forward(inputs, expected_width):
            assert inputs["input_ids"].shape[1] == expected_width
            mask = inputs["attention_mask"].bool()
            backbone = {
                "backbone_features": features[:, -expected_width:],
                "backbone_attention_mask": mask,
            }
            result = Gr00tN1d7ActionHead.process_backbone_output(head, backbone)
            assert result["backbone_attention_mask"] is mask
            assert result["backbone_features"].shape == (2, expected_width, features.shape[2])
            return result["backbone_features"]

        for width in (152, 149, 156):
            valid = columns >= torch.tensor([[156 - width], [159 - width]])
            batch = collated_batch(width)
            actual = run(vl, lambda: trainer.training_step(lambda inputs: forward(inputs, 156), batch),
                         valid, image, width)
            ref = expected[width]
            for out, ref_out in zip(actual[:3], ref[:3], strict=True):
                torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-6)
            for grads, ref_grads in zip(actual[3:5], ref[3:5], strict=True):
                assert grads.keys() == ref_grads.keys() and ref_grads
                for name in grads:
                    torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-6)
            assert torch.equal(actual[-1], ref[-1])
            if compile_vl and width == 152:
                first_graph_count = len(graphs)
                assert first_graph_count > 0
            if compile_vl and width != 152:
                assert len(graphs) == first_graph_count

        # Calls without a collator width still run the standard VL path.
        torch.manual_seed(123)
        plain = plain_forward(features)
        torch.manual_seed(123)
        torch.testing.assert_close(vl(features), plain, rtol=1e-4, atol=1e-6)
        if compile_vl:
            assert len(graphs) > first_graph_count

        # Evaluation runs after training in the same Trainer and keeps the collator width.
        baseline.eval()
        vl.eval()
        trainer.label_names = []
        trainer.can_return_loss = False
        trainer.use_cpu_amp = False
        _, actual_eval, _ = trainer.prediction_step(
            lambda inputs: {"features": forward(inputs, 152)},
            collated_batch(152),
            prediction_loss_only=False,
            ignore_keys=[],
        )
        with torch.no_grad():
            expected_eval = plain_forward(features[:, -152:])
        torch.testing.assert_close(actual_eval, expected_eval, rtol=1e-4, atol=1e-6)
