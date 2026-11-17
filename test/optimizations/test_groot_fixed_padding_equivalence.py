"""Does left padding (``groot.data.pad_156``) change the model's numerics?

The baseline action head calls ``vl_self_attention(backbone_features)`` without an attention mask,
while ``AlternateVLDiT`` receives ``encoder_attention_mask`` and ignores masked positions. These
tests use tiny random modules to measure both stages separately.
"""

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


def run(vl, dit, batch, *, padded, use_vl_self_attention):
    prefix = "padded_" if padded else ""
    features = batch[f"{prefix}features"]
    if use_vl_self_attention:
        features = vl(features)
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
    vl, dit = modules
    plain = run(vl, dit, batch, padded=False, use_vl_self_attention=False)
    padded = run(vl, dit, batch, padded=True, use_vl_self_attention=False)
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


def test_pad_156_changes_loss_and_gradients_through_vl_self_attention(modules, batch):
    vl, dit = modules
    plain = run(vl, dit, batch, padded=False, use_vl_self_attention=True)
    padded = run(vl, dit, batch, padded=True, use_vl_self_attention=True)
    relative = (padded - plain).norm() / plain.norm()
    assert relative > 1e-3, f"relative output difference {relative:.2e}"
    plain_grads = parameter_gradients(vl, plain.square().mean())
    padded_grads = parameter_gradients(vl, padded.square().mean())
    # Global norms: per-parameter ratios are dominated by parameters whose gradient is near zero.
    difference = torch.cat([(padded_grads[n] - g).flatten() for n, g in plain_grads.items()]).norm()
    reference = torch.cat([g.flatten() for g in plain_grads.values()]).norm()
    relative = difference / reference
    assert relative > 1e-3, f"expected vl_self_attention gradients to change, relative {relative:.2e}"
