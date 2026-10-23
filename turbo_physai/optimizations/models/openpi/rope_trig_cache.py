# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional cached exact trig and native arithmetic for the confirmed RoPE graphs."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import statistics


def _install_runtime():
    if os.environ.get('OPENPI_TORCH_KERNEL_MODE','off')!='gate':return
    import torch
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from .torch_runtime.rope_candidate import build,expression_key
    from .torch_runtime.rope_trig import precompute
    original=CachingAutotuner.run
    root=Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT']);rank=int(os.environ['LOCAL_RANK'])
    state={'rank':rank,'steps':0,'precompute_calls':0,'calls':{},'configs':{}}
    atexit.register(lambda:(root/f'rope_hits_rank{rank}.json').write_text(json.dumps(state,indent=2)))
    cache={};kernels={};fnkeys={};observed=set()
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    original_forward = PI0Pytorch.forward

    def forward_with_step_cache(self, *args, **kwargs):
        # Official OpenPI calls model.forward once per training step. This
        # reproduces the patch runner's before_forward cache lifetime without
        # requiring that runner or modifying OpenPI source.
        cache.clear()
        state['steps'] += 1
        return original_forward(self, *args, **kwargs)

    PI0Pytorch.forward = forward_with_step_cache
    def trig(freq):
        assert freq.dtype==torch.float32 and freq.is_contiguous() and freq.numel()==4005888
        key=(freq.data_ptr(),freq._version)
        if key not in cache:
            c,s=torch.empty_like(freq),torch.empty_like(freq)
            precompute[(triton.cdiv(freq.numel(),1024),)](freq,c,s,N=freq.numel(),B=1024,num_warps=4,enable_fp_fusion=True)
            cache[key]=(freq,c,s);state['precompute_calls']+=1
        return cache[key][1:]
    def timed(fn):
        for _ in range(2):fn()
        events=[]
        for _ in range(7):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();fn();b.record();events.append((a,b))
        torch.cuda.synchronize()
        return statistics.median(a.elapsed_time(b) for a,b in events)
    def grid(args,cfg):
        ny=triton.cdiv(args[-2],cfg['YBLOCK']);nx=triton.cdiv(args[-1],cfg['XBLOCK'])
        return (nx,min(ny,65535),triton.cdiv(ny,65535)) if ny>65535 else (nx,ny)
    def run(self,*args,stream,**kw):
        name=self.inductor_meta.get('kernel_name','')
        if 'cos' in name and 'sin' in name and name not in observed:
            observed.add(name)
            row={'name':name,'source':self.fn.src,'signature':self.triton_meta['signature'],
                 'arguments':[{'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'numel':x.numel()} if isinstance(x,torch.Tensor) else x for x in args]}
            (root/f'rope_seen_rank{rank}_{name}.json').write_text(json.dumps(row,indent=2))
            print('OPENPI_ROPE_SEEN '+json.dumps({k:v for k,v in row.items() if k not in ['source','signature']}),flush=True)
        pointers=len(args)-2
        freq_index=1 if len(args)==5 else 2
        matching=('cos' in name and 'sin' in name and 'cat' in name
                  and len(args) in [5,10] and isinstance(args[-1],int) and isinstance(args[-2],int)
                  and args[0].dtype==torch.bfloat16 and args[freq_index].dtype==torch.float32 and args[freq_index].numel()==4005888
                  and args[0].numel()>1000000
                  and (args[-2:] in [(250368,256),(65536,978),(247808,256),(30976,256)]))
        if not matching:return original(self,*args,stream=stream,**kw)
        if id(self) not in fnkeys:fnkeys[id(self)]=expression_key(self.fn.src)
        key=fnkeys[id(self)]
        if key not in kernels:
            original(self,*args,stream=stream,**kw)
            assert self.device_props.cc=='gfx938' and self.device_props.multi_processor_count==64
            assert torch.cuda.current_stream().cuda_stream==stream
            c,s=trig(args[freq_index])
            kernel,digest=build(self.fn.src,root/f'rope_generated_rank{rank}')
            native_cfg=self.launchers[0].config
            outputs=[i for i,n in enumerate(self.triton_meta['signature']) if n.startswith(('out_ptr','in_out_ptr'))]
            golden=[args[i].clone() for i in outputs]
            report={'key':key,'name':name,'source':self.fn.src,'source_sha256':hashlib.sha256(self.fn.src.encode()).hexdigest(),
                    'candidate_sha256':digest,'native_config':dict(kwargs=native_cfg.kwargs,num_warps=native_cfg.num_warps),
                    'enable_fp_fusion':self.triton_meta.get('enable_fp_fusion',True),'shape':list(args[0].shape),
                    'input_stride':list(args[0].stride()),'output_shape':[list(args[i].shape) for i in outputs],'output_stride':[list(args[i].stride()) for i in outputs],'results':[]}
            def launch(cfg):
                return kernel[grid(args,cfg)](*args[:pointers],c,s,*args[pointers:],**cfg,num_stages=1,enable_fp_fusion=report['enable_fp_fusion'])
            options=[dict(YBLOCK=native_cfg.kwargs['YBLOCK'],XBLOCK=native_cfg.kwargs['XBLOCK'],num_warps=native_cfg.num_warps)]
            options += [dict(YBLOCK=y,XBLOCK=x,num_warps=w) for y,x,w in [(1,256,4),(4,256,4),(8,128,4),(8,256,4),(16,128,4),(4,512,4),(4,1024,8),(16,64,4),(32,64,4),(32,128,8),(16,256,8)]]
            report['native_ms']=timed(lambda:original(self,*args,stream=stream,**kw))
            policy=json.loads((Path(__file__).parent/'torch_runtime'/'rope_policy.json').read_text())['cases'].get(key)
            if policy is not None:options=[policy['config']]
            modules={}
            for side in ([policy['fma_side']] if policy is not None else [None,'right','left']):
             kernel,digest=build(self.fn.src,root/f'rope_generated_rank{rank}',side)
             modules[str(side)]=(kernel,digest)
             for cfg in options:
                try:
                    launch(cfg)
                    neq=sum(int(torch.count_nonzero(g.view(torch.int16)!=args[i].view(torch.int16))) for g,i in zip(golden,outputs))
                    row={'config':cfg,'unequal':neq,'fma_side':str(side)}
                    if neq==0:row['ms']=timed(lambda:launch(cfg))
                    report['results'].append(row)
                except Exception as e:report['results'].append({'config':cfg,'fma_side':str(side),'error':str(e)[:1000]})
            exact=[x for x in report['results'] if x.get('unequal')==0]
            best=min(exact,key=lambda x:x['ms']) if exact else None
            report['best']=best
            (root/f'rope_micro_rank{rank}_{key[:12]}.json').write_text(json.dumps(report,indent=2))
            print('OPENPI_ROPE_SCREEN '+json.dumps({k:v for k,v in report.items() if k not in ['source','results']}),flush=True)
            if best is not None and (policy is not None or best['ms']<report['native_ms']*.98):
                kernel,digest=modules[best['fma_side']]
                report['candidate_sha256']=digest
                kernels[key]=(kernel,best['config'],report['enable_fp_fusion'])
            else:
                kernels[key]=None
            state['configs'][key]=best['config'] if best else None;state['calls'][key]=0
            for i,g in zip(outputs,golden):args[i].copy_(g)
            return
        if kernels[key] is None:return original(self,*args,stream=stream,**kw)
        kernel,cfg,fusion=kernels[key]
        c,s=trig(args[freq_index]);state['calls'][key]+=1
        return kernel[grid(args,cfg)](*args[:pointers],c,s,*args[pointers:],**cfg,num_stages=1,enable_fp_fusion=fusion)
    CachingAutotuner.run=run


from .torch_runtime.installation import wrap_runtime_install

def wrap_rope_trig_cache(original, options):
    del options
    return wrap_runtime_install(original, "openpi.rope_trig_cache", _install_runtime)
