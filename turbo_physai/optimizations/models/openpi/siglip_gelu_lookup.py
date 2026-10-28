# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in exact SigLIP GELU tables. Native calls remain fallback."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import statistics

def wrap_run(original):
 import torch,triton
 from .torch_runtime.siglip_lookup import build
 from .gemma_gate_lookup import expression_key
 root=Path(os.environ['OPENPI_TORCH_TIMING_OUTPUT']);rank=int(os.environ['LOCAL_RANK'])
 states={};keys={}
 atexit.register(lambda:(root/f'siglip_hits_rank{rank}.json').write_text(json.dumps([{k:v for k,v in s.items() if k not in ['module','table']} for s in states.values()],indent=2)))
 def run(self,*args,stream,**kw):
  name=self.inductor_meta.get('kernel_name','')
  match=(('triton_poi_fused_gelu_view_' in name or 'triton_poi_fused_gelu_gelu_backward_view_' in name)
         and len(args)==3 and isinstance(args[-1],int) and args[-1]>1000000
         and all(x.dtype==torch.bfloat16 for x in args[:2]))
  if not match:return original(self,*args,stream=stream,**kw)
  if id(self) not in keys:keys[id(self)]=expression_key(self.fn.src,self.triton_meta.get('enable_fp_fusion',True))
  key=keys[id(self)]
  if key not in states:
   backward='backward' in name
   saved=args[0].clone() if backward else None
   original(self,*args,stream=stream,**kw)
   golden=args[0 if backward else 1].clone()
   assert self.device_props.cc=='gfx938' and self.device_props.multi_processor_count==64
   module=build(self.fn.src,root/f'siglip_sources_rank{rank}')
   table=torch.empty(65536,dtype=torch.float32 if backward else torch.bfloat16,device=args[0].device)
   fusion=self.triton_meta.get('enable_fp_fusion',True)
   module['table'][(64,)](table,XBLOCK=1024,num_warps=4,enable_fp_fusion=fusion)
   codes=torch.arange(65536,device=args[0].device,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
   factors=torch.tensor([1.,-.3125,.00390625,-128.],device=args[0].device,dtype=torch.bfloat16).repeat(16384)
   small=[factors.clone(),codes] if backward else [codes,torch.empty_like(codes)]
   module['reference_small'][(64,)](*small,65536,XBLOCK=1024,num_warps=4,enable_fp_fusion=fusion)
   small_ref=small[0 if backward else 1].clone()
   if backward:small[0].copy_(factors)
   module['candidate_small'][(64,)](*small,65536,table,XBLOCK=1024,num_warps=4,enable_fp_fusion=fusion)
   neq=int(torch.count_nonzero(small_ref.view(torch.int16)!=small[0 if backward else 1].view(torch.int16)))
   report={'key':key,'name':name,'source_sha256':hashlib.sha256(self.fn.src.encode()).hexdigest(),
           'candidate_sha256':module['digest'],'backward':backward,'all_bf16_unequal':neq,'results':[]}
   assert neq==0,report
   def restore():
    if saved is not None:args[0].copy_(saved)
   def timed(f):
    for _ in range(2):restore();f()
    pairs=[]
    for _ in range(9):
     restore();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
     a.record();f();b.record();pairs.append((a,b))
    torch.cuda.synchronize();return statistics.median(a.elapsed_time(b) for a,b in pairs)
   report['native_ms']=timed(lambda:original(self,*args,stream=stream,**kw))
   policy=json.loads((Path(__file__).parent / 'torch_runtime' / 'siglip_policy.json').read_text())['cases'].get(key)
   options=[(policy['block'],policy['warps'])] if policy else []
   for block,warps in options:
    def launch():module['kernel'][(triton.cdiv(args[-1],block),)](*args,table,XBLOCK=block,num_warps=warps,enable_fp_fusion=fusion)
    restore();launch();neq=int(torch.count_nonzero(golden.view(torch.int16)!=args[0 if backward else 1].view(torch.int16)))
    row={'block':block,'warps':warps,'real_unequal':neq}
    if neq==0:row['ms']=timed(launch)
    report['results'].append(row)
   exact=[x for x in report['results'] if x.get('real_unequal')==0 and (policy is not None or x['ms']<report['native_ms']*.98)]
   best=min(exact,key=lambda x:x['ms']) if exact else None
   report['best']=best
   (root/f'siglip_micro_rank{rank}_{key[:12]}.json').write_text(json.dumps(report,indent=2))
   print('OPENPI_SIGLIP_SCREEN '+json.dumps(report),flush=True)
   states[key]={'key':key,'backward':backward,'module':module,'table':table,'choice':best,'fusion':fusion,'replaced_calls':0}
   args[0 if backward else 1].copy_(golden);return
  s=states[key];best=s['choice']
  if best is None:return original(self,*args,stream=stream,**kw)
  s['replaced_calls']+=1
  return s['module']['kernel'][(triton.cdiv(args[-1],best['block']),)](*args,s['table'],XBLOCK=best['block'],num_warps=best['warps'],enable_fp_fusion=s['fusion'])
 return run
