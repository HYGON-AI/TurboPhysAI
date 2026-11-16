import functools
import re
import types
import torch
from torch import nn

_PATTERN = re.compile(r'blocks\.\d+\.(?:attn[12]\.to_[qkv]|attn[12]\.to_out\.0|ffn\.net\.0\.proj|ffn\.net\.2)$')
# Keep action_embedder on the original nn.Linear path: its addmm fusion changes
# BF16 rounding relative to the model's matmul-then-bias computation.
_ROOTS = {'patch_embedding_mlp', 'proj_out', 'action_proj_out'}

def linear_forward(self, input):
    flat = input.reshape(-1, input.shape[-1])
    output = torch.mm(flat, self.weight.t()) if self.bias is None else torch.addmm(self.bias, flat, self.weight.t())
    return output.reshape(*input.shape[:-1], self.out_features)

def install_linears(model):
    for name, module in model.named_modules():
        if type(module) is nn.Linear and (name in _ROOTS or _PATTERN.fullmatch(name)):
            module.forward = types.MethodType(linear_forward, module)

def linear_wrapper(original, options):
    if options:raise ValueError('linear_addmm accepts no options')
    @functools.wraps(original)
    def constructor(self, *args, **kwargs):
        original(self, *args, **kwargs)
        install_linears(self)
    return constructor
