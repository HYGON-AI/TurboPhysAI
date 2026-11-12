import torch.nn as nn

from turbo_physai.optimizations.models.lingbot_va.linear import install_linears, linear_forward


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embedder = nn.Linear(4, 4)
        self.patch_embedding_mlp = nn.Linear(4, 4)
        self.proj_out = nn.Linear(4, 4)


def test_action_embedder_keeps_original_linear_forward():
    model = _LinearModel()

    install_linears(model)

    assert model.action_embedder.forward.__func__ is nn.Linear.forward
    assert model.patch_embedding_mlp.forward.__func__ is linear_forward
    assert model.proj_out.forward.__func__ is linear_forward
