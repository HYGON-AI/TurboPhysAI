# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in exact-expression gate policy; retain native calls for every other case."""
import ast
import atexit
import hashlib
import json
import os
from pathlib import Path

_states={}
_registered=False
_tuner_keys={}


def expression_key(source,fusion):
    tree=ast.parse(source)
    tree.body[0].name='kernel'
    return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()+':fp'+str(int(fusion))


def dispatch(tuner,original,args,stream,kwargs):
    global _registered
    import torch
    from .torch_runtime.gate_lookup import build
    bundle=Path(__file__).resolve().parent/'torch_runtime'
    root=Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT'])
    rank=int(os.environ['LOCAL_RANK'])
    source=tuner.fn.src
    fusion=tuner.triton_meta.get('enable_fp_fusion',True)
    identity=id(tuner)
    if identity not in _tuner_keys:
        _tuner_keys[identity]=expression_key(source,fusion)
    key=_tuner_keys[identity]
    if not _registered:
        def save():
            rows=[{k:v for k,v in s.items() if k not in {'module','table','tables'}} for s in _states.values() if s is not None]
            (root/f'gate_hits_rank{rank}.json').write_text(json.dumps(rows,indent=2))
        atexit.register(save)
        _registered=True
    if key not in _states:
        result=original(tuner,*args,stream=stream,**kwargs)
        policy=json.loads((bundle/'gate_policy.json').read_text())
        choice=policy['cases'].get(key)
        if choice is None:
            _states[key]=None
            (root/f'gate_unmatched_rank{rank}_{key[:12]}.json').write_text(json.dumps({'key':key,'source':source},indent=2))
            return result
        assert tuner.device_props.cc=='gfx938' and tuner.device_props.multi_processor_count==64
        assert all(x.dtype==torch.bfloat16 and x.is_contiguous() and x.numel()==507510784 for x in args if isinstance(x,torch.Tensor))
        cfg=tuner.launchers[0].config
        meta={'source':source,'signature':tuner.triton_meta['signature']}
        module=build(meta,root/f'gate_installed_sources_rank{rank}',choice.get('variant','full'))
        assert module['kind']==choice['kind']
        tables=[torch.empty(65536,device=args[0].device,dtype=torch.float32) for _ in range(module['tables'])]
        module['table'][(64,)](*tables,XBLOCK=1024,num_warps=4,num_stages=1,enable_fp_fusion=fusion)
        table=(torch.stack(tables,dim=1).contiguous() if module['kind']=='backward' else tables[0])
        _states[key]={'key':key,'kind':choice['kind'],'backend':choice['backend'],
            'config':choice['config'],'original_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'variant':choice.get('variant','full'),
            'lookup_source_sha256':module['source_sha256'],'enable_fp_fusion':fusion,
            'native_config':{'kwargs':cfg.kwargs,'num_warps':cfg.num_warps},
            'module':module,'tables':tables,'table':table,'replaced_calls':0}
        (root/f'gate_runtime_actual_rank{rank}.py').write_text(Path(__file__).read_text())
        installs=[{k:v for k,v in s.items() if k not in {'module','table','tables'}} for s in _states.values() if s is not None]
        (root/f'gate_install_rank{rank}.json').write_text(json.dumps(installs,indent=2))
        print('OPENPI_GATE_INSTALL '+json.dumps({'rank':rank,'kind':choice['kind'],'backend':choice['backend'],'key':key}),flush=True)
        return result
    state=_states[key]
    if state is None:return original(tuner,*args,stream=stream,**kwargs)
    assert torch.cuda.current_stream().cuda_stream==stream
    cfg=state['config']
    if state['backend']=='hip_shared':
        from .torch_runtime.gate_shared import call
        call(state['kind'],args,state['table'],stream,cfg)
    elif state['backend']=='triton_lookup':
        state['module']['persistent' if cfg['persistent'] else 'direct'][(cfg['grid'],)](
            *args,*state['tables'],XBLOCK=cfg['XBLOCK'],num_warps=cfg['num_warps'],
            num_stages=1,enable_fp_fusion=fusion)
    else:raise AssertionError(state['backend'])
    state['replaced_calls']+=1


def _install_runtime():
    from .torch_runtime.kernel_hooks import install
    install()


from .torch_runtime.installation import wrap_runtime_install

def wrap_gemma_gate_lookup(original, options):
    del options
    return wrap_runtime_install(original, "openpi.gemma_gate_lookup", _install_runtime)
