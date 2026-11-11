# Adapted from Robbyant/lingbot-va 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb, Apache-2.0.
import functools
import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy

def save_matmul(ctx, op, *args, **kwargs):
    if op in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default):
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE

def checkpoint_factory(original, options):
    if set(options)-{'tail_layers', 'sac_layer_interval'}:raise ValueError('Unknown checkpoint options')
    tail=options.get('tail_layers',0)
    interval=options.get('sac_layer_interval',1)
    if isinstance(interval,bool) or not isinstance(interval,int) or interval<1:raise ValueError('sac_layer_interval must be positive integer')
    if isinstance(tail,bool) or not isinstance(tail,int) or tail<0:raise ValueError('tail_layers must be nonnegative integer')
    @functools.wraps(original)
    def apply_ac(model):
        if tail>len(model.blocks):raise ValueError('tail_layers exceeds block count')
        for i in range(len(model.blocks)-tail):
            kwargs={'preserve_rng_state':False}
            if i % interval == 0:
                kwargs['context_fn']=functools.partial(create_selective_checkpoint_contexts,save_matmul)
            model.blocks[i]=checkpoint_wrapper(model.blocks[i],**kwargs)
    return apply_ac

def residency_factory(original, options):
    if set(options)-{'reshard_after_forward'}:raise ValueError('Unknown FSDP options')
    reshard=options.get('reshard_after_forward',False)
    if type(reshard) is not bool:raise ValueError('reshard_after_forward must be bool')
    @functools.wraps(original)
    def shard_model(model,param_dtype=torch.bfloat16,reduce_dtype=torch.float32):
        policy=MixedPrecisionPolicy(param_dtype=param_dtype,reduce_dtype=reduce_dtype,cast_forward_inputs=False)
        config={'mp_policy':policy,'reshard_after_forward':reshard}
        for block in model.blocks:
            for module in (block.attn1,block.attn2,block.ffn,block):fully_shard(module,**config)
        fully_shard(model,**config)
        return model
    return shard_model
