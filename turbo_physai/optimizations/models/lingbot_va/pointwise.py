# Adapted from Robbyant/lingbot-va 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb, Apache-2.0.
import functools
import os
import torch
from einops import rearrange

def modulation(norm,x,scale,shift):
    return (norm(x.float())*(1.+scale)+shift).type_as(x)

def gated_residual(x,branch,gate):
    return (x.float()+branch.float()*gate).type_as(x)

def make_forward(modulate,residual,cross_normalize=None):
    if cross_normalize is None:
        cross_normalize=lambda norm,value: norm(value.float()).type_as(value)
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = modulate(self.norm1, hidden_states, scale_msa, shift_msa)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = residual(hidden_states, attn_output, gate_msa)

        # 2. Cross-attention
        norm_hidden_states = cross_normalize(self.norm2,hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = modulate(self.norm3, hidden_states, c_scale_msa, c_shift_msa)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = residual(hidden_states, ff_output, c_gate_msa)
        return hidden_states
    return forward

def pointwise_wrapper(original,options):
    if os.getenv('TURBO_PHYSAI_DISABLE_TORCH_COMPILE','0')=='1':return original
    if set(options)-{'cross_norm'}:raise ValueError('Unknown pointwise_compile options')
    enabled=options.get('cross_norm',False)
    if type(enabled) is not bool:raise ValueError('cross_norm must be bool')
    modulate=torch.compile(modulation,backend='inductor',fullgraph=True,dynamic=True)
    residual=torch.compile(gated_residual,backend='inductor',fullgraph=True,dynamic=True)
    normalize=None
    if enabled:
        from .norms import layer_norm_cast, layer_adapter
        normalize=layer_adapter(torch.compile(layer_norm_cast,backend='inductor',fullgraph=True,dynamic=True))
    return functools.wraps(original)(make_forward(modulate,residual,normalize))
