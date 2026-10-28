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


