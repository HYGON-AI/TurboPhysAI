# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Construct tables from Torch's own FP32 gate expression, with no approximation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

HEADER=('import triton\nimport triton.language as tl\n'
        'from torch._inductor.runtime import triton_helpers\n'
        'from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n')


def build(meta,root,variant='full'):
    tree=ast.parse(meta['source'])
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    names=[n.arg for n in fn.args.args]
    if names==['in_ptr0','in_ptr1','out_ptr0','xnumel','XBLOCK']:
        xpointer='in_ptr0';xvar='tmp0';targets=['tmp15'];kind='forward'
    else:
        assert names==['in_out_ptr0','in_ptr0','in_ptr1','out_ptr0','xnumel','XBLOCK']
        xpointer='in_ptr1';xvar='tmp1';targets=['tmp16','tmp30'];kind='backward'
    assigns={n.targets[0].id:n for n in fn.body if isinstance(n,ast.Assign)
             and isinstance(n.targets[0],ast.Name)}
    if variant=='tanh':
        targets=[name for name,n in assigns.items() if 'libdevice.tanh(' in ast.unparse(n.value)]
        assert len(targets)==1
    else:assert variant in {'full','full_cg'}
    needed=set()
    def require(name):
        if name in needed:return
        needed.add(name)
        if name==xvar:return
        assert name in assigns,name
        for n in ast.walk(assigns[name].value):
            if isinstance(n,ast.Name) and n.id.startswith('tmp'):require(n.id)
    for name in targets:require(name)
    table_body=['idx=tl.program_id(0)*XBLOCK+tl.arange(0,XBLOCK)',
                f'{xvar}=idx.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)']
    for n in fn.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if name in needed and name!=xvar:table_body.append(ast.unparse(n))
    table_body += [f'tl.store(table{i}+idx,{name})' for i,name in enumerate(targets)]
    table_name='openpi_torch_gate_table_'+kind+'_'+variant
    table_source='@triton.jit\ndef '+table_name+'('+','.join('table'+str(i) for i in range(len(targets)))+',XBLOCK:tl.constexpr):\n'
    table_source+='\n'.join('    '+s for s in table_body)+'\n'
    candidate=copy.deepcopy(fn)
    candidate.name+='_openpi_lookup_'+variant
    candidate.decorator_list=[ast.Name(id='triton.jit')]
    for i in range(len(targets)):
        candidate.args.args.insert(-1,ast.arg(arg='table'+str(i)))
    body=[]
    for n in candidate.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
            name=n.targets[0].id
            if variant=='tanh':
                if name==targets[0]:
                    body.extend(ast.parse(f'lookup_index=tl.load({xpointer}+x0,None).to(tl.uint16,bitcast=True).to(tl.int32)\n{name}=tl.load(table0+lookup_index)').body)
                else:body.append(n)
                continue
            if name==xvar:
                body.extend(ast.parse(f'lookup_index=tl.load({xpointer}+x0,None).to(tl.uint16,bitcast=True).to(tl.int32)').body)
                body.extend(ast.parse('\n'.join(f'{t}=tl.load(table{i}+lookup_index)' for i,t in enumerate(targets))).body)
                continue
            if name in needed:continue
        body.append(n)
    candidate.body=body
    if variant=='full_cg':
        for node in ast.walk(candidate):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                    and isinstance(node.func.value,ast.Name) and node.func.value.id=='tl'
                    and node.func.attr=='load'):
                pointer=ast.unparse(node.args[0])
                node.keywords.append(ast.keyword(arg='cache_modifier',
                    value=ast.Constant(value='.ca' if 'table' in pointer else '.cg')))
    ast.fix_missing_locations(candidate)
    direct=ast.unparse(candidate)
    persistent=copy.deepcopy(candidate)
    persistent.name+='_persistent'
    start=next(i for i,n in enumerate(persistent.body) if isinstance(n,ast.Assign)
               and n.targets[0].id=='xoffset')
    loop=ast.parse('for xoffset in range(tl.program_id(0)*XBLOCK,xnumel,tl.num_programs(0)*XBLOCK):\n    pass').body[0]
    loop.body=persistent.body[start+1:]
    persistent.body=persistent.body[:start]+[loop]
    ast.fix_missing_locations(persistent)
    # A short original kernel exercises every BF16 encoding with test factors.
    exhaustive=copy.deepcopy(fn)
    exhaustive.name+='_exhaustive_reference'
    exhaustive.decorator_list=[ast.Name(id='triton.jit')]
    exhaustive.body[0].value=ast.Constant(value=65536)
    small=copy.deepcopy(candidate)
    small.name+='_exhaustive_candidate'
    small.body[0].value=ast.Constant(value=65536)
    ast.fix_missing_locations(exhaustive);ast.fix_missing_locations(small)
    generated=HEADER+table_source+'\n'+direct+'\n'+ast.unparse(persistent)+'\n'+ast.unparse(exhaustive)+'\n'+ast.unparse(small)+'\n'
    digest=hashlib.sha256(generated.encode()).hexdigest()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/f'lookup_{digest}.py';path.write_text(generated)
    spec=importlib.util.spec_from_file_location('openpi_lookup_'+digest,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return {'table':getattr(module,table_name),'direct':getattr(module,candidate.name),
            'persistent':getattr(module,persistent.name),'exhaustive_reference':getattr(module,exhaustive.name),
            'exhaustive_candidate':getattr(module,small.name),'tables':len(targets),
            'xpointer':xpointer,'kind':kind,'source_sha256':digest}


def screen(tuner,original,args,stream,kwargs,meta,root,variant_override=None):
    import os
    import torch
    import triton
    import statistics
    variant=variant_override or ('tanh' if os.environ.get('OPENPI_TORCH_TANH_SCREEN')=='1' else 'full')
    module=build(meta,root/'lookup_sources',variant)
    assert torch.cuda.current_stream().cuda_stream==stream
    # The native call has already produced golden outputs; save them and the
    # overwritten up projection separately before any repeated invocation.
    signature=list(meta['signature'])
    outputs=[i for i,n in enumerate(signature) if n.startswith(('out_ptr','in_out_ptr'))]
    # Input snapshots are supplied by the dispatcher before the native call.
    inputs=kwargs.pop('_openpi_saved_inputs')
    golden=[args[i].clone() for i in outputs]
    tables=[torch.empty(65536,device=args[0].device,dtype=torch.float32) for _ in range(module['tables'])]
    launch=dict(num_warps=4,num_stages=1,enable_fp_fusion=meta['enable_fp_fusion'])
    module['table'][(64,)](*tables,XBLOCK=1024,**launch)
    report={'name':meta['name'],'original_source_sha256':meta['source_sha256'],
            'lookup_source_sha256':module['source_sha256'],'kind':module['kind'],'variant':variant,'results':[]}
    small_args=[]
    factors=torch.tensor([1.,-.3125,.00390625,-128.],device=args[0].device,dtype=torch.bfloat16).repeat(16384)
    codes=torch.arange(65536,device=args[0].device,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    for i,n in enumerate(signature):
        if i>=len(args):break
        if isinstance(args[i],torch.Tensor):small_args.append(codes.clone() if n==module['xpointer'] else factors.clone())
        else:small_args.append(65536)
    module['exhaustive_reference'][(64,)](*small_args,XBLOCK=1024,**launch)
    small_golden=[small_args[i].clone() for i in outputs]
    for i in inputs:small_args[i].copy_(factors)
    module['exhaustive_candidate'][(64,)](*small_args,*tables,XBLOCK=1024,**launch)
    report['exhaustive_bf16_output_unequal']=[int(torch.count_nonzero(a.view(torch.int16)!=small_args[i].view(torch.int16))) for i,a in zip(outputs,small_golden)]
    def restore():
        for i,t in inputs.items():args[i].copy_(t)
    def native():original(tuner,*args,stream=stream,**kwargs)
    def bench(fn):
        for _ in range(3):restore();fn()
        pairs=[]
        for _ in range(15):
            restore();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record();fn();b.record();pairs.append((a,b))
        torch.cuda.synchronize()
        return statistics.median(a.elapsed_time(b) for a,b in pairs)
    try:
        report['native_ms_before']=bench(native)
        for persistent,block,grid in [(False,1024,None),(False,2048,None),(False,4096,None),
                                      (True,4096,512),(True,8192,512)]:
            cfg={'persistent':persistent,'XBLOCK':block,'num_warps':4,'grid':grid or triton.cdiv(507510784,block)}
            def candidate():module['persistent' if persistent else 'direct'][(cfg['grid'],)](*args,*tables,XBLOCK=block,**launch)
            restore();candidate()
            cfg['real_output_bitwise_unequal']=[int(torch.count_nonzero(a.view(torch.int16)!=args[i].view(torch.int16))) for i,a in zip(outputs,golden)]
            cfg['bitwise_equal']=all(n==0 for n in cfg['real_output_bitwise_unequal']) and all(n==0 for n in report['exhaustive_bf16_output_unequal'])
            if cfg['bitwise_equal']:cfg['milliseconds']=bench(candidate)
            report['results'].append(cfg)
        report['native_ms_after']=bench(native)
    finally:
        for i,a in zip(outputs,golden):args[i].copy_(a)
    (root/f'gate_lookup_{variant}_{meta["source_sha256"][:12]}.json').write_text(json.dumps(report,indent=2))
    print('OPENPI_GATE_LOOKUP '+json.dumps(report),flush=True)
    if os.environ.get('OPENPI_TORCH_SHARED_SCREEN') == '1':
        from gate_shared import screen as shared_screen
        shared_screen(module['kind'],args,tables,stream,meta,root,small_args,small_golden,
                      golden,outputs,inputs,native)
    if variant=='tanh' and variant_override is None:
        screen(tuner,original,args,stream,dict(kwargs,_openpi_saved_inputs=inputs),
               meta,root,variant_override='full_cg')
