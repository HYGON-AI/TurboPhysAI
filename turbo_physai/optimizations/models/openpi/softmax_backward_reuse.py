# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Reuse intermediates only in the audited full-row Softmax backward expression."""
import ast
import atexit
import hashlib
import json
import os
from pathlib import Path


def expression_key(source):
    tree = ast.parse(source)
    tree.body[0].name = 'kernel'
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _install_runtime():
    if os.environ.get('OPENPI_TORCH_KERNEL_MODE', 'off') == 'off':
        return
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    original = CachingAutotuner.run
    root = Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT'])
    rank = int(os.environ['LOCAL_RANK'])
    policy = json.loads((Path(__file__).parent/'torch_runtime'/'softmax_reuse_policy.json').read_text())
    states = {}

    def public():
        return [{k: v for k, v in row.items() if k != 'kernel'}
                for row in states.values() if row is not None]

    atexit.register(lambda: (root/f'reuse_hits_rank{rank}.json').write_text(
        json.dumps(public(), indent=2)))

    def run(self, *args, stream, **kwargs):
        name = self.inductor_meta.get('kernel_name', '')
        if ('softmax_backward_data' not in name or len(args) != 9
                or not all(isinstance(x, int) for x in args[-2:])
                or args[-2:] != (250368, 978)):
            return original(self, *args, stream=stream, **kwargs)
        identity = id(self)
        if identity not in states:
            result = original(self, *args, stream=stream, **kwargs)
            states[identity] = None
            cfg = self.launchers[0].config
            key = expression_key(self.fn.src)
            seen = dict(key=key, source=self.fn.src, kwargs=cfg.kwargs,
                        warps=cfg.num_warps, stages=cfg.num_stages,
                        enable_fp_fusion=self.triton_meta.get('enable_fp_fusion', True))
            (root/f'reuse_seen_rank{rank}_{key[:12]}.json').write_text(json.dumps(seen, indent=2))
            if (key != policy['expression_key']
                    or dict(kwargs=cfg.kwargs,num_warps=cfg.num_warps) not in policy['native_configs']
                    or cfg.num_warps != policy['num_warps']
                    or seen['enable_fp_fusion'] != policy['enable_fp_fusion']
                    or self.device_props.cc != 'gfx938'
                    or self.device_props.multi_processor_count != 64):
                return result
            from .torch_runtime.softmax_backward_candidate import build
            kernel, source = build(self.fn.src, root/f'reuse_sources_rank{rank}')
            states[identity] = dict(key=key, kernel=kernel, kwargs=dict(policy['kwargs']), native_kwargs=dict(cfg.kwargs),
                                    num_warps=policy['num_warps'], num_stages=cfg.num_stages,
                                    enable_fp_fusion=seen['enable_fp_fusion'],
                                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                    runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                    replaced_calls=0)
            (root/f'reuse_install_rank{rank}.json').write_text(json.dumps(public(), indent=2))
            print('OPENPI_SOFTMAX_REUSE_INSTALL '+json.dumps(public()), flush=True)
            # Check the real two outputs once, without allocating GPU snapshots.
            def digest(tensor):
                raw=tensor.detach().cpu().contiguous().reshape(-1).view(__import__('torch').uint8).numpy()
                return hashlib.sha256(raw).hexdigest()
            before=[digest(args[index]) for index in [5,6]]
            state=states[identity]
            state['kernel'][((250368+state['kwargs']['XBLOCK']-1)//state['kwargs']['XBLOCK'],)](
                *args, **state['kwargs'], num_warps=state['num_warps'],
                num_stages=state['num_stages'], enable_fp_fusion=state['enable_fp_fusion'])
            after=[digest(args[index]) for index in [5,6]]
            state['micro_bitwise_equal']=before==after
            state['micro_output_sha256']=after
            (root/f'reuse_micro_rank{rank}.json').write_text(json.dumps(
                dict(before=before,after=after,bitwise_equal=before==after),indent=2))
            assert before==after, 'Full-row Softmax reuse changed real outputs'
            return result
        state = states[identity]
        if state is None:
            return original(self, *args, stream=stream, **kwargs)
        import torch
        import triton
        assert torch.cuda.current_stream().cuda_stream == stream
        state['replaced_calls'] += 1
        return state['kernel'][(triton.cdiv(250368, state['kwargs']['XBLOCK']),)](
            *args, **state['kwargs'], num_warps=state['num_warps'],
            num_stages=state['num_stages'], enable_fp_fusion=state['enable_fp_fusion'])

    CachingAutotuner.run = run


from .torch_runtime.installation import wrap_runtime_install

def wrap_softmax_backward_reuse(original, options):
    del options
    return wrap_runtime_install(original, "openpi.softmax_backward_reuse", _install_runtime)
