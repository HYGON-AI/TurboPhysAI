# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Exact per-encoding SigLIP GELU / derivative tables from the original AST."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
HEADER='import triton\nimport triton.language as tl\nfrom torch._inductor.runtime import triton_helpers\nfrom torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n'

def build(source,directory):
 tree=ast.parse(source);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef))
 names=[a.arg for a in fn.args.args];assert names in [['in_ptr0','out_ptr0','xnumel','XBLOCK'],['in_out_ptr0','in_ptr0','xnumel','XBLOCK']]
 backward=names[0]=='in_out_ptr0';xp='in_ptr0'
 assignments={n.targets[0].id:n for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
 loads={name:n for name,n in assignments.items() if 'tl.load('+xp+' ' in ast.unparse(n.value)}
 assert len(loads)==1,loads.keys()
 xvar=next(iter(loads));store=next(n.value for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='tl.store')
 target=store.args[1].id
 if backward:
  last=assignments[target].value
  assert isinstance(last,ast.Call) and ast.unparse(last.func).endswith('.to')
  prod=assignments[last.func.value.id].value
  assert isinstance(prod,ast.BinOp) and isinstance(prod.op,ast.Mult)
  target=prod.right.id
 needed=set()
 def require(name):
  if name in needed:return
  needed.add(name)
  if name==xvar:return
  assert name in assignments,name
  for n in ast.walk(assignments[name].value):
   if isinstance(n,ast.Name) and n.id.startswith('tmp'):require(n.id)
 require(target)
 body=[ast.parse('idx=tl.program_id(0)*XBLOCK+tl.arange(0,XBLOCK)').body[0],
       ast.parse(xvar+'=idx.to(tl.uint16).to(tl.bfloat16,bitcast=True).to(tl.float32)').body[0]]
 for n in fn.body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in needed and n.targets[0].id!=xvar:body.append(copy.deepcopy(n))
 body+=ast.parse(f'tl.store(table+idx,{target})').body
 table=ast.parse('@triton.jit\ndef make_table(table,XBLOCK:tl.constexpr):\n pass').body[0];table.body=body
 candidate=copy.deepcopy(fn);candidate.name+='_openpi_siglip_lookup';candidate.decorator_list=[ast.Name(id='triton.jit')]
 candidate.args.args.insert(-1,ast.arg(arg='table'))
 cb=[]
 xload=copy.deepcopy(loads[xvar].value)
 assert isinstance(xload,ast.Call) and ast.unparse(xload.func).endswith('.to')
 raw=xload.func.value
 assert ast.unparse(raw.func)=='tl.load'
 replacement=ast.parse(f"code=({ast.unparse(raw)}).to(tl.uint16,bitcast=True).to(tl.int32)\n{target}=tl.load(table+code)").body
 for n in candidate.body:
  if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name):
   if n.targets[0].id==xvar:cb+=replacement;continue
   if n.targets[0].id in needed:continue
  cb.append(n)
 candidate.body=cb
 # Exhaustive helper makes the same native arithmetic operate on compact arrays.
 exhaustive=copy.deepcopy(fn);exhaustive.name+='_exhaustive';exhaustive.decorator_list=[ast.Name(id='triton.jit')]
 small=copy.deepcopy(candidate);small.name+='_exhaustive'
 for variant in [exhaustive,small]:
  variant.body[0].value=ast.Constant(value=65536)
  for n in ast.walk(variant):
   if isinstance(n,ast.Call) and ast.unparse(n.func) in ['tl.load','tl.store']:
    ptr=n.args[0]
    if isinstance(ptr,ast.BinOp) and isinstance(ptr.left,ast.Name) and ptr.left.id in ['in_ptr0','out_ptr0','in_out_ptr0']:
     ptr.right=ast.Name(id='xindex',ctx=ast.Load())
 for n in [table,candidate,exhaustive,small]:ast.fix_missing_locations(n)
 code=HEADER+'\n'.join(ast.unparse(n) for n in [table,candidate,exhaustive,small])+'\n'
 digest=hashlib.sha256(code.encode()).hexdigest();root=Path(directory);root.mkdir(parents=True,exist_ok=True)
 p=root/f'siglip_{digest}.py';p.write_text(code);spec=importlib.util.spec_from_file_location('openpi_siglip_'+digest,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return {'table':getattr(mod,table.name),'kernel':getattr(mod,candidate.name),'reference_small':getattr(mod,exhaustive.name),
         'candidate_small':getattr(mod,small.name),'backward':backward,'digest':digest,'outputs':[0 if backward else 1]}
