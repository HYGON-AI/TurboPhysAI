"""FlexAttention tile configuration only; original business BlockMask is reused."""
import functools
import torch
from einops import rearrange

def flex_call(q,k,v,*,block_mask,kernel_options):
    from modules.model import FlexAttnFunc
    return FlexAttnFunc.flex_attn(q,k,v,block_mask=block_mask,kernel_options=kernel_options)

_DEFAULT={'BLOCK_M':128,'BLOCK_N':32,'BLOCK_M1':16,'BLOCK_N1':64,'BLOCK_M2':64,'BLOCK_N2':32,'fwd_num_warps':4,'bwd_num_warps':4,'fwd_num_stages':1,'bwd_num_stages':1}

def attention_wrapper(original, options):
    options=dict(options)
    cross_fa=options.pop('cross_fa',False)
    if type(cross_fa) is not bool:raise ValueError('cross_fa must be bool')
    if set(options)-set(_DEFAULT):raise ValueError('Unknown FlexAttention option')
    config=dict(_DEFAULT,**options)
    if any(type(v) is not int or v<=0 for v in config.values()):raise ValueError('Flex options must be positive integers')
    @functools.wraps(original)
    def forward(self,query,key,value,dtype=torch.bfloat16):
        from modules.model import FlexAttnFunc
        # The upstream packs logical samples into the sequence dimension.
        # Preserve its supported physical layout and fall back otherwise.
        if query.ndim!=4 or key.ndim!=4 or value.ndim!=4 or any(x.shape[0]!=1 for x in (query,key,value)):
            return original(self,query,key,value,dtype=dtype)
        assert dtype in (torch.float16,torch.bfloat16)
        tensors=[rearrange(x[0],'s n d -> 1 n s d') for x in (query,key,value)]
        q,k,v=[x if x.dtype in (torch.float16,torch.bfloat16) else x.to(dtype) for x in tensors]
        q=q.to(v.dtype);k=k.to(v.dtype)
        mask=FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask
        result=flex_call(q,k,v,block_mask=mask,kernel_options=dict(config))
        return rearrange(result,'b n s d -> b s n d')
    return cross_fa_wrapper(forward,{}) if cross_fa else forward

def mask_metadata_wrapper(original, options):
    @functools.wraps(original)
    def init_mask(latent_shape,action_shape,padded_length,chunk_size,window_size,patch_size,device):
        from modules.model import FlexAttnFunc
        result=original(latent_shape,action_shape,padded_length,chunk_size,window_size,patch_size,device)
        b,_,f,h,w=latent_shape
        ab,_,af,ah,aw=action_shape
        valid=2*((f//patch_size[0])*(h//patch_size[1])*(w//patch_size[2])+af*ah*aw)
        # Derived from original un-packed shapes and the original mask builder.
        FlexAttnFunc._lingbot_cross_shape=(b,ab,valid,valid+padded_length,patch_size[0])
        return result
    return init_mask

def cross_fa_wrapper(original, options):
    if options:raise ValueError('cross_fa accepts no options')
    @functools.wraps(original)
    def forward(self,query,key,value,dtype=torch.bfloat16):
        from modules.model import FlexAttnFunc,flash_attn_func
        context=getattr(FlexAttnFunc,'_lingbot_cross_shape',None)
        if (not self.is_cross or context is None or context[0:2]!=(1,1) or context[4]!=1
            or query.ndim!=4 or key.ndim!=4 or value.ndim!=4
            or query.shape[0]!=1 or key.shape[0]!=1 or value.shape[0]!=1
            or key.shape[1]!=512 or value.shape!=key.shape
            or query.shape[1]!=context[3] or query.shape[2:]!=key.shape[2:]
            or context[2]<=0 or context[2]>query.shape[1]):
            return original(self,query,key,value,dtype=dtype)
        assert dtype in (torch.float16,torch.bfloat16)
        q,k,v=[x if x.dtype in (torch.float16,torch.bfloat16) else x.to(dtype) for x in (query,key,value)]
        q=q.to(v.dtype);k=k.to(v.dtype)
        out=flash_attn_func(q[:,:context[2]],k,v,dropout_p=0.0,softmax_scale=None,causal=False)
        return torch.nn.functional.pad(out,(0,0,0,0,0,query.shape[1]-context[2]))
    return forward
