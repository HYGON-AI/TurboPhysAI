"""Opt-in Group boundary. Installation is exclusively an environment preparation step."""
import functools
import hashlib
import json
from pathlib import Path

REVISION=3796128826
MANIFEST_SHA256='a73970ead7f810932150d1c17443c3c24c56670217658117cb1ec9d8fe4262c5'

def validate_backend(root=None):
    import torch
    import triton
    root=Path(root) if root is not None else Path(torch.__file__).resolve().parent
    assets=Path(__file__).parent/'flex_assets'
    path=assets/'manifest.json'
    if not path.is_file():raise RuntimeError('Flex template assets missing')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA256:raise RuntimeError('Asset manifest hash mismatch')
    m=json.loads(raw)
    if (m['revision']!=REVISION or m['torch_version']!=torch.__version__ or
        m['torch_git_version']!=torch.version.git_version or m['hip_version']!=torch.version.hip or
        m['triton_version']!=triton.__version__):
        raise RuntimeError('Torch/HIP/Triton build mismatch')
    for name,digest in m['original_hashes'].items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Original Torch hash mismatch: '+name)
    for name,digest in m['assets'].items():
        p=assets/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Template asset hash mismatch: '+name)
    return m

def compatible(context):
    from turbo_physai.engine.contracts import CompatibilityResult
    from .compatibility import baseline
    original=baseline(context)
    if not original.compatible:return original
    try:validate_backend()
    except (RuntimeError,OSError,KeyError,ValueError) as error:
        return CompatibilityResult(False,reason=str(error))
    return CompatibilityResult(True,expected={'backend_revision':REVISION},actual={'backend_revision':REVISION},reason='Pinned backend and original model verified')

def supported(q,k,v,*,block_mask,kernel_options):
    import torch
    if not torch.version.hip or any(t.device.type!='cuda' or t.dtype!=torch.bfloat16 or t.ndim!=4 for t in (q,k,v)):
        return False
    if k.shape!=v.shape or any(t.shape[0]!=1 or t.shape[1]!=24 or t.shape[3]!=128 for t in (q,k,v)):
        return False
    if any(t.shape[2]<=0 or t.shape[2]%128 for t in (q,k,v)):return False
    if any(t.device!=q.device or t.stride(-1)!=1 for t in (q,k,v)):return False
    return getattr(torch.cuda.get_device_properties(q.device),'gcnArchName','').split(':')[0]=='gfx936'

def backend_wrapper(original,options):
    if set(options)-{'fast_bf16'}:raise ValueError('Unknown Flex backend option')
    fast=options.get('fast_bf16',False)
    if type(fast) is not bool:raise ValueError('fast_bf16 must be bool')
    @functools.wraps(original)
    def call(q,k,v,*,block_mask,kernel_options):
        if any(key in kernel_options for key in ('LINGBOT_VA_TEMPLATE_OPT','LINGBOT_VA_BACKEND_REV','LINGBOT_FAST_BF16')):
            raise ValueError('Backend selector options are owned by the Group')
        config=dict(kernel_options)
        config.update(LINGBOT_VA_TEMPLATE_OPT=True,LINGBOT_VA_BACKEND_REV=REVISION,LINGBOT_FAST_BF16=fast)
        return original(q,k,v,block_mask=block_mask,kernel_options=config)
    return call


def template_wrapper(original,options):
    @functools.wraps(original)
    def call(self,*args,**kwargs):
        if self.name not in ('flex_attention','flex_attention_backward') or not kwargs.get('LINGBOT_VA_TEMPLATE_OPT',False):
            return original(self,*args,**kwargs)
        if kwargs.get('LINGBOT_VA_BACKEND_REV')!=REVISION:raise RuntimeError('Template revision mismatch')
        from .flex_assets import implementation as opt
        kw=dict(kwargs);kw.pop('LINGBOT_VA_TEMPLATE_OPT')
        kw.setdefault('LINGBOT_FAST_BF16',False)
        q,k,v=kw['input_nodes'][:3]
        backward=self.name=='flex_attention_backward'
        kw['IS_DIVISIBLE']=opt._guard_sequence_divisibility(q.get_size()[2],k.get_size()[2],kw,backward=backward)
        selected=opt.flex_attention_backward_template if backward else opt.flex_attention_template
        if backward:
            tuned=opt._guard_gfx936_bwd_tuning(q,k,v,kw)
            kw['USE_GFX936_BWD_TUNING']=tuned
            kw.pop('HIP_BACKEND_KPACK',None)
            if tuned:kw.update(waves_per_eu=4,HIP_BACKEND_KPACK=2)
        return original(selected,*args,**kw)
    return call


def compile_options_wrapper(original,options):
    @functools.wraps(original)
    def call(self,*args,**kwargs):
        result=original(self,*args,**kwargs)
        config=self.inductor_meta.get('config_args',{})
        if config.get('LINGBOT_VA_BACKEND_REV')==REVISION and config.get('HIP_BACKEND_KPACK') is not None:
            result=dict(result);result['kpack']=config['HIP_BACKEND_KPACK']
        return result
    return call
