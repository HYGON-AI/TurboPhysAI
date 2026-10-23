# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in observation and local replacement of confirmed Inductor kernels."""
import json
import os
from pathlib import Path


def install():
    mode = os.environ.get('OPENPI_TORCH_KERNEL_MODE', 'off')
    if mode == 'off':
        return
    assert mode in {'capture', 'softmax', 'shadow', 'backward','gate_capture','gate_screen','gate_lookup_screen','gate'}
    import torch
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    original = CachingAutotuner.run
    seen = set()
    root = Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT'])
    rank = int(os.environ['LOCAL_RANK'])
    if (mode == 'gate_lookup_screen' and os.environ.get('OPENPI_TORCH_SHARED_SCREEN') != '1'
            and os.environ.get('OPENPI_TORCH_TANH_SCREEN') != '1'):
        from mlp_screen import install as install_mlp_screen
        install_mlp_screen()

    def run(self, *args, stream, **kwargs):
        name = self.inductor_meta.get('kernel_name', '')
        if mode.startswith('gate'):
            target = ('gelu' in name and any(isinstance(x, int) and x == 507510784 for x in args))
            if target:
                from .gate_candidate import dispatch
                return dispatch(self, original, args, stream, kwargs, mode)
            return original(self, *args, stream=stream, **kwargs)
        target = ('softmax' in name and len(args) >= 2
                  and all(isinstance(x, int) for x in args[-2:])
                  and args[-2:] == (250368, 978))
        if not target:
            return original(self, *args, stream=stream, **kwargs)
        if mode == 'backward':
            from .softmax_backward_candidate import dispatch
            return dispatch(self, original, args, stream, kwargs, mode)
        if mode in {'softmax', 'shadow'}:
            from .softmax_candidate import dispatch
            return dispatch(self, original, args, stream, kwargs, mode)
        result = original(self, *args, stream=stream, **kwargs)
        if name not in seen:
            seen.add(name)
            cfg = self.launchers[0].config
            metadata = {
                'rank': rank, 'name': name, 'source': self.fn.src,
                'signature': self.triton_meta['signature'],
                'config': {'kwargs': cfg.kwargs, 'num_warps': cfg.num_warps,
                           'num_stages': cfg.num_stages},
                'enable_fp_fusion': self.triton_meta.get('enable_fp_fusion', True),
                'arguments': [dict(shape=list(x.shape), stride=list(x.stride()),
                                   dtype=str(x.dtype)) if isinstance(x, torch.Tensor)
                              else x for x in args],
            }
            (root / f'kernel_rank{rank}_{name}.json').write_text(json.dumps(metadata, indent=2))
            if rank == 0:
                # One real call per expression; profiling/capture is never a QPS result.
                torch.save([x.detach().cpu() if isinstance(x, torch.Tensor) else x
                            for x in args], root / (name + '.pt'))
            print('OPENPI_KERNEL_CAPTURE ' + json.dumps({k: v for k, v in metadata.items()
                  if k not in {'source', 'signature'}}), flush=True)
        return result

    CachingAutotuner.run = run
    print('OPENPI_KERNEL_MODE ' + mode, flush=True)
