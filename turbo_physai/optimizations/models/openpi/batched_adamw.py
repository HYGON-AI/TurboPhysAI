# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in batched scheduling; tensor arithmetic and BF16 rounding are unchanged."""
import atexit
import json
import os
from pathlib import Path


def _install_runtime():
    if os.environ.get('OPENPI_TORCH_KERNEL_MODE') != 'gate':
        return
    import torch
    from .torch_runtime.adam_batched import launch, prepare
    original = torch._fused_adamw_
    root = Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT'])
    rank = int(os.environ['LOCAL_RANK'])
    precision = bool(os.environ.get('OPENPI_TORCH_PRECISION'))
    state = dict(rank=rank, calls=0, tensor_calls=0, elements=0, enabled=True,
                 shadow_tensors=0, shadow_elements=0, shadow_unequal=[0, 0, 0],
                 batch_launches=0, metadata_uploads=0)
    atexit.register(lambda: (root/f'adam_hits_rank{rank}.json').write_text(json.dumps(state, indent=2)))
    cached = {}

    def eligible(row):
        p, g, m, v, s = row
        return (p.dtype == torch.bfloat16 and p.numel() >= 1048576 and p.numel() % 16 == 0
                and p.device.type == 'cuda' and all(
                    t.is_contiguous() and t.dtype == p.dtype and t.shape == p.shape and t.device == p.device
                    for t in [p, g, m, v])
                and all(t.data_ptr() % 32 == 0 for t in [p, g, m, v])
                and s.dtype == torch.float32 and s.numel() == 1 and s.device == p.device)

    def config(row):
        n = row[0].numel()
        return (8, 1, 512, 256) if n < 8000000 else (
            (16, 2, 1024, 256) if n < 100000000 else (16, 2, 4096, 256))

    def update(rows, cfg, kw, cache_key=None):
        pointers = tuple(tuple(t.data_ptr() for t in row) for row in rows)
        stored = cached.get(cache_key) if cache_key is not None else None
        if stored is None or stored[0] != pointers:
            metadata, scales = prepare(rows, rows[0][0].device)
            # The caller owns current tensor allocations; the table is reused only
            # after matching all live pointers again. Do not retain old gradients.
            stored = (pointers, metadata, scales)
            if cache_key is not None:
                cached[cache_key] = stored
            state['metadata_uploads'] += 1
        launch(stored[1], stored[2], len(rows), cfg,
               kw['lr'], kw['beta1'], kw['beta2'], kw['weight_decay'], kw['eps'])
        state['batch_launches'] += 1

    def patched(ps, gs, ms, vs, maxs, ss, **kw):
        supported = (kw.get('amsgrad') is False and kw.get('maximize') is False
                     and kw.get('grad_scale') is None and kw.get('found_inf') is None and not maxs
                     and all(isinstance(kw[k], (float, int)) for k in ['lr', 'beta1', 'beta2', 'weight_decay', 'eps']))
        if not supported:
            return original(ps, gs, ms, vs, maxs, ss, **kw)
        state['calls'] += 1
        groups, rest = {}, []
        for i, row in enumerate(zip(ps, gs, ms, vs, ss)):
            if eligible(row):
                groups.setdefault(config(row), []).append(row)
            else:
                rest.append(i)
        if rest:
            original(*[[xs[i] for i in rest] for xs in [ps, gs, ms, vs]], [], [ss[i] for i in rest], **kw)
        for cfg, rows in groups.items():
            if precision:
                # Bound diagnostic memory while exercising the same independent
                # tensor kernel, including its prepared coefficients.
                for offset in range(0, len(rows), 4):
                    chunk = rows[offset:offset+4]
                    shadows = [[t.clone() for t in [r[0], r[2], r[3]]] for r in chunk]
                    original([s[0] for s in shadows], [r[1] for r in chunk],
                             [s[1] for s in shadows], [s[2] for s in shadows], [],
                             [r[4] for r in chunk], **kw)
                    update(chunk, cfg, kw)
                    for row, shadow in zip(chunk, shadows):
                        unequal = [int(torch.count_nonzero(a.view(torch.int16) != b.view(torch.int16)))
                                   for a, b in zip(shadow, [row[0], row[2], row[3]])]
                        state['shadow_tensors'] += 1
                        state['shadow_elements'] += row[0].numel()
                        state['shadow_unequal'] = [a+b for a, b in zip(state['shadow_unequal'], unequal)]
                        assert not any(unequal), ('Batched AdamW parameter/moment mismatch', unequal)
            else:
                update(rows, cfg, kw, cfg)
            state['tensor_calls'] += len(rows)
            state['elements'] += sum(r[0].numel() for r in rows)
    torch._fused_adamw_ = patched


from .torch_runtime.installation import wrap_runtime_install

def wrap_batched_adamw(original, options):
    del options
    return wrap_runtime_install(original, "openpi.batched_adamw", _install_runtime)
