from ....engine.definitions import group, wrap

TIMESTEP = group(
    'lingbot_va.timestep_dedup',
    wrap('modules.model.WanTransformer3DModel._time_embed',
         'turbo_physai.optimizations.models.lingbot_va.timestep.timestep_wrapper'),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline',
)
ROTARY = group(
    'lingbot_va.rotary_compile',
    wrap('modules.model.WanAttention.forward',
         'turbo_physai.optimizations.models.lingbot_va.rotary.rotary_wrapper'),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline',
)
LINEAR = group('lingbot_va.linear_addmm',
    wrap('modules.model.WanTransformer3DModel.__init__','turbo_physai.optimizations.models.lingbot_va.linear.linear_wrapper'),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline')
POINTWISE = group('lingbot_va.pointwise_compile',
    wrap('modules.model.WanTransformerBlock.forward','turbo_physai.optimizations.models.lingbot_va.pointwise.pointwise_wrapper'),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline')
ATTENTION = group('lingbot_va.attention_backend',
    wrap('modules.model.FlexAttnFunc.init_mask','turbo_physai.optimizations.models.lingbot_va.attention.mask_metadata_wrapper'),
    wrap('modules.model.FlexAttnFunc.forward','turbo_physai.optimizations.models.lingbot_va.attention.attention_wrapper'),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline')
CHECKPOINT = group('lingbot_va.selective_checkpoint',
    wrap('distributed.fsdp.apply_ac','turbo_physai.optimizations.models.lingbot_va.fsdp.checkpoint_factory',aliases=('train.apply_ac',)),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline')
RESIDENCY = group('lingbot_va.fsdp_residency',
    wrap('distributed.fsdp.shard_model','turbo_physai.optimizations.models.lingbot_va.fsdp.residency_factory',aliases=('train.shard_model',)),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline')
SYNC = group('lingbot_va.train_sync',
    wrap('train.Trainer.train','turbo_physai.optimizations.models.lingbot_va.train_sync.sync_wrapper'),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.compatibility.baseline')

FLEX_TEMPLATE=group('lingbot_va.flex_template_backend',
    wrap('turbo_physai.optimizations.models.lingbot_va.attention.flex_call',
         'turbo_physai.optimizations.models.lingbot_va.group_backend.backend_wrapper',
         runtime_condition='turbo_physai.optimizations.models.lingbot_va.group_backend.supported'),
    wrap('torch._inductor.select_algorithm.TritonTemplate.maybe_append_choice','turbo_physai.optimizations.models.lingbot_va.group_backend.template_wrapper'),
    wrap('torch._inductor.runtime.triton_heuristics.CachingAutotuner._create_compile_options','turbo_physai.optimizations.models.lingbot_va.group_backend.compile_options_wrapper'),
    depends_on=('lingbot_va.attention_backend',),
    compatibility_check='turbo_physai.optimizations.models.lingbot_va.group_backend.compatible')
