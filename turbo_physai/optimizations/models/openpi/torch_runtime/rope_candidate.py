# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Hoist repeated FP32 trig from a known, unchanged Inductor pointwise graph."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
HEADER='import triton\nimport triton.language as tl\nfrom torch._inductor.runtime import triton_helpers\nfrom torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n'

def expression_key(source):
 t=ast.parse(source);t.body[0].name='kernel'
 return hashlib.sha256(ast.dump(t,include_attributes=False).encode()).hexdigest()

def build(source,root,fma_side=None):
 t=ast.parse(source);fn=next(x for x in t.body if isinstance(x,ast.FunctionDef))
 assert [x.arg for x in fn.args.args][-4:]==['ynumel','xnumel','YBLOCK','XBLOCK']
 assert len(fn.args.args) in [7,12]
 assignments={n.targets[0].id:n.value for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
 assert assignments['ynumel'].value in [250368,65536,247808,30976,2560,320] or assignments['xnumel'].value==2048
 count=0
 for n in fn.body:
  if not (isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)):continue
  call=n.value
  if ast.unparse(call.func) not in ['tl_math.sin','tl_math.cos']:continue
  assert len(call.args)==1 and isinstance(call.args[0],ast.Name)
  load=copy.deepcopy(assignments[call.args[0].id])
  assert isinstance(load,ast.Call) and ast.unparse(load.func)=='tl.load'
  assert isinstance(load.args[0],ast.BinOp) and ast.unparse(load.args[0].left) in ['in_ptr1','in_ptr2']
  trig=call.func.attr
  load.args[0].left=ast.Name(id=trig+'_ptr',ctx=ast.Load())
  # sin(0)=0, cos(0)=1 for masked input loads. NaN paths still use real data.
  load.keywords=[x for x in load.keywords if x.arg!='other']+[ast.keyword(arg='other',value=ast.Constant(value=0. if trig=='sin' else 1.))]
  n.value=load;count+=1
 assert count>=2
 if fma_side is not None:
  for store in [n.value for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='tl.store']:
   dest=next(n for n in fn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==store.args[1].id)
   if not (isinstance(dest.value,ast.BinOp) and isinstance(dest.value.op,ast.Add)):continue
   lhs,rhs=dest.value.left,dest.value.right
   selected=lhs if fma_side=='left' else rhs
   other=rhs if fma_side=='left' else lhs
   mul=assignments[selected.id]
   if isinstance(mul,ast.BinOp) and isinstance(mul.op,ast.Mult):
    dest.value=ast.Call(func=ast.Attribute(value=ast.Name(id='tl',ctx=ast.Load()),attr='fma',ctx=ast.Load()),args=[copy.deepcopy(mul.left),copy.deepcopy(mul.right),copy.deepcopy(other)],keywords=[])
 fn.name+='_openpi_trig_reuse'
 pointers=len(fn.args.args)-4
 fn.args.args.insert(pointers,ast.arg(arg='cos_ptr'));fn.args.args.insert(pointers+1,ast.arg(arg='sin_ptr'))
 fn.decorator_list=[ast.Name(id='triton.jit',ctx=ast.Load())]
 ast.fix_missing_locations(fn)
 code=HEADER+ast.unparse(fn)+'\n'
 digest=hashlib.sha256(code.encode()).hexdigest()
 root=Path(root);root.mkdir(exist_ok=True,parents=True);p=root/f'rope_{digest}.py';p.write_text(code)
 spec=importlib.util.spec_from_file_location('openpi_rope_'+digest,p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 return getattr(mod,fn.name),digest
