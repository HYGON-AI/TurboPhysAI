# Adapted from Robbyant/lingbot-va, commit 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb.
# Original copyright: 2024-2025 The Robbyant Team Authors. All rights reserved.
import functools
import os
import torch


def rotary_real(x, freqs):
    pairs = x.to(torch.float64).unflatten(-1, (-1, 2))
    real, imag = pairs[..., 0], pairs[..., 1]
    c, s = freqs.real, freqs.imag
    return torch.stack((real * c - imag * s, real * s + imag * c), dim=-1).flatten(-2).to(x.dtype)


def make_forward(rotate, normalize=None):
    if normalize is None:
        normalize = lambda norm, value: norm(value)
    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = normalize(self.norm_q, query)
        query = query.unflatten(2, (self.heads, -1))
        key = normalize(self.norm_k, key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            apply_rotary_emb = rotate
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    return forward


def rotary_wrapper(original, options):
    if os.getenv('TURBO_PHYSAI_DISABLE_TORCH_COMPILE', '0') == '1':
        return original
    if set(options)-{'qk_norm'}:raise ValueError('Unknown rotary_compile options')
    enabled=options.get('qk_norm',False)
    if type(enabled) is not bool:raise ValueError('qk_norm must be bool')
    rotate = torch.compile(rotary_real, backend='inductor', fullgraph=True, dynamic=True)
    normalize=None
    if enabled:
        from .norms import rms_norm, rms_adapter
        normalize=rms_adapter(torch.compile(rms_norm,backend='inductor',fullgraph=True,dynamic=True))
    return functools.wraps(original)(make_forward(rotate,normalize))
