# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Shape-scoped Torch gate observation and arithmetic-preserving scheduling."""
import hashlib
import json
import os
from pathlib import Path

_seen = set()


def dispatch(tuner, original, args, stream, kwargs, mode):
    if mode == 'gate':
        from ..gemma_gate_lookup import dispatch as runtime_dispatch
        return runtime_dispatch(tuner,original,args,stream,kwargs)
    import torch
    name = tuner.inductor_meta['kernel_name']
    key = hashlib.sha256(tuner.fn.src.encode()).hexdigest()
    rank = int(os.environ['LOCAL_RANK'])
    inputs = {}
    if mode in {'gate_screen','gate_lookup_screen'} and rank == 0 and key not in _seen:
        inputs = {i:args[i].clone() for i,n in enumerate(tuner.triton_meta['signature'])
                  if n.startswith('in_out_ptr')}
    result = original(tuner, *args, stream=stream, **kwargs)
    if key not in _seen:
        _seen.add(key)
        rank = int(os.environ['LOCAL_RANK'])
        root = Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT'])
        cfg = tuner.launchers[0].config
        meta = {'name':name, 'rank':rank, 'source_sha256':key,
                'source':tuner.fn.src, 'signature':tuner.triton_meta['signature'],
                'config':{'kwargs':cfg.kwargs,'num_warps':cfg.num_warps,'num_stages':cfg.num_stages},
                'enable_fp_fusion':tuner.triton_meta.get('enable_fp_fusion',True),
                'arguments':[dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),
                                  elements=x.numel()) if isinstance(x,torch.Tensor) else x for x in args]}
        (root/f'gate_rank{rank}_{key[:12]}.json').write_text(json.dumps(meta,indent=2))
        print('OPENPI_GATE_CAPTURE '+json.dumps({'rank':rank,'name':name,'config':meta['config']}),flush=True)
        if mode == 'gate_screen' and rank == 0:
            from .gate_screen import screen
            screen(tuner,original,args,stream,kwargs,meta,root,inputs)
        if mode == 'gate_lookup_screen' and rank == 0:
            from .gate_lookup import screen
            screen(tuner,original,args,stream,dict(kwargs,_openpi_saved_inputs=inputs),meta,root)
    return result
