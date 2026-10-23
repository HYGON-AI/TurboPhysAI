# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Only native R0_BLOCK reduction grouping can be selected."""
import atexit
import json
import os
from pathlib import Path
import statistics


def _install_runtime():
    if os.environ.get('OPENPI_TORCH_KERNEL_MODE') != 'gate':
        return
    import torch
    import triton
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from .torch_runtime.softmax_schedule import build
    original = CachingAutotuner.run
    root = Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT']); rank = int(os.environ['LOCAL_RANK'])
    states = {}
    atexit.register(lambda: (root/f'softmax_schedule_hits_rank{rank}.json').write_text(json.dumps(
        [{k:v for k,v in s.items() if k != 'kernels'} for s in states.values()], indent=2)))

    def run(self, *args, stream, **kw):
        name = self.inductor_meta.get('kernel_name', '')
        if ('prepare_softmax_online' not in name or len(args) != 7 or args[-2:] != (250368, 978)):
            return original(self, *args, stream=stream, **kw)
        identity = id(self)
        if identity not in states:
            original(self, *args, stream=stream, **kw)
            cfg = self.launchers[0].config
            assert self.device_props.cc == 'gfx938' and self.device_props.multi_processor_count == 64
            assert torch.cuda.current_stream().cuda_stream == stream
            kernels, digest = build(self.fn.src, root/f'softmax_schedule_sources_rank{rank}')
            golden = [a.clone() for a in args[2:5]]
            fusion = self.triton_meta.get('enable_fp_fusion', True)
            report = dict(source=self.fn.src, digest=digest, native_config=dict(kwargs=cfg.kwargs, warps=cfg.num_warps), results=[])

            def launch(choice):
                opts = dict(XBLOCK=choice['x'], R0_BLOCK=cfg.kwargs['R0_BLOCK'], num_warps=choice['w'], num_stages=1, enable_fp_fusion=fusion)
                if choice['kind'] in ['schedule', 'oneexp']:
                    kernels[3 if choice['kind'] == 'oneexp' else 0][(triton.cdiv(250368, choice['x']),)](*args, **opts)
                else:
                    kernels[1][(triton.cdiv(250368, choice['x']),)](*args, **opts)
                    out = choice['out']
                    kernels[2][(triton.cdiv(250368, out['x']),)](*args, XBLOCK=out['x'], R0_BLOCK=1024,
                        num_warps=out['w'], num_stages=1, enable_fp_fusion=fusion)

            def timed(fn):
                for _ in range(2): fn()
                pairs = []
                for _ in range(5):
                    a,b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); fn(); b.record(); pairs.append((a,b))
                torch.cuda.synchronize()
                return statistics.median(a.elapsed_time(b) for a,b in pairs)

            report['native_ms'] = timed(lambda: original(self, *args, stream=stream, **kw))
            options = [dict(kind='schedule', x=x, w=w) for x,w in [(8,4),(16,4),(32,4),(64,4),(64,8),(128,8),(32,2),(16,2),(128,16)]]
            options += [dict(kind='split', x=x, w=w, out=dict(x=ox,w=ow)) for x,w,ox,ow in [(32,8,4,8),(64,8,4,8),(32,4,8,8),(64,4,8,8)]]
            options = [dict(kind='oneexp', x=16, w=4)]
            for c in options:
                row=dict(choice=c)
                try:
                    launch(c)
                    unequal=[int(torch.count_nonzero(g != a)) for g,a in zip(golden,args[2:5])]
                    row['unequal']=unequal
                    if not any(unequal):row['ms']=timed(lambda:launch(c))
                except Exception as e:
                    row['error']=str(e)[-1200:]
                report['results'].append(row)
                (root/f'softmax_schedule_micro_rank{rank}.json').write_text(json.dumps(report,indent=2))
            exact=[r for r in report['results'] if 'ms' in r]
            best=min(exact,key=lambda r:r['ms']) if exact else None
            report['best']=best
            (root/f'softmax_schedule_micro_rank{rank}.json').write_text(json.dumps(report,indent=2))
            print('OPENPI_SOFTMAX_SCHEDULE '+json.dumps({k:v for k,v in report.items() if k not in ['source','results']}),flush=True)
            selected=best['choice'] if best and best['ms'] < report['native_ms']*.98 else None
            states[identity]=dict(kernels=kernels, choice=selected, fusion=fusion, rblock=cfg.kwargs['R0_BLOCK'], calls=0)
            for a,g in zip(args[2:5],golden):a.copy_(g)
            return
        s=states[identity];c=s['choice']
        if c is None:return original(self,*args,stream=stream,**kw)
        opts=dict(XBLOCK=c['x'],R0_BLOCK=s['rblock'],num_warps=c['w'],num_stages=1,enable_fp_fusion=s['fusion'])
        s['calls']+=1
        if c['kind'] in ['schedule','oneexp']:
            return s['kernels'][3 if c['kind']=='oneexp' else 0][(triton.cdiv(250368,c['x']),)](*args,**opts)
        s['kernels'][1][(triton.cdiv(250368,c['x']),)](*args,**opts)
        o=c['out']
        return s['kernels'][2][(triton.cdiv(250368,o['x']),)](*args,XBLOCK=o['x'],R0_BLOCK=1024,
            num_warps=o['w'],num_stages=1,enable_fp_fusion=s['fusion'])

    CachingAutotuner.run = run


from .torch_runtime.installation import wrap_runtime_install

def wrap_softmax_forward_schedule(original, options):
    del options
    return wrap_runtime_install(original, "openpi.softmax_forward_schedule", _install_runtime)
