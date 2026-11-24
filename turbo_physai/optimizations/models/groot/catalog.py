"""groot optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.
"""

from turbo_physai import group, replace, wrap


TRAINING_CONFIG = group(
    "groot.compat.training_config",
    wrap(
        target="gr00t.experiment.experiment.run",
        aliases=("gr00t.experiment.launch_finetune.run",),
        replacement="turbo_physai.optimizations.models.groot.compat.training_config_wrapper",
    ),
)

OPENCV_VIDEO = group(
    "groot.compat.opencv_video",
    wrap(
        target="gr00t.utils.video_utils.get_frames_by_indices",
        aliases=("gr00t.data.dataset.lerobot_episode_loader.get_frames_by_indices",),
        replacement="turbo_physai.optimizations.models.groot.compat.opencv_video_wrapper",
    ),
)


DIT_HIP_MATH_SDPA = group(
    "groot.dit.hip_math_sdpa",
    wrap(
        target="gr00t.model.modules.dit._sdpa_context",
        replacement="turbo_physai.optimizations.models.groot.replacements.dit_sdpa_context_wrapper",
    ),
)

_QWEN = "transformers.models.qwen3_vl.modeling_qwen3_vl"
_POS = "turbo_physai.optimizations.models.groot.qwen_position"

QWEN_FAST_POS = group(
    "groot.qwen.fast_pos_embed_interpolate",
    replace(
        target=f"{_QWEN}.Qwen3VLVisionModel.fast_pos_embed_interpolate",
        replacement=f"{_POS}.fast_pos_embed_interpolate",
    ),
)

QWEN_ROT_POS = group(
    "groot.qwen.rot_pos_emb",
    replace(
        target=f"{_QWEN}.Qwen3VLVisionModel.rot_pos_emb",
        replacement=f"{_POS}.rot_pos_emb",
    ),
)

QWEN_ROPE = group(
    "groot.qwen.get_rope_index",
    wrap(
        target=f"{_QWEN}.Qwen3VLModel.get_rope_index",
        replacement=f"{_POS}.rope_index_wrapper",
    ),
)

FLASH_DISABLE = group(
    "groot.qwen.flash_attn_dynamo_disable",
    wrap(
        target="transformers.modeling_utils.AttentionInterface._global_mapping.flash_attention_2",
        aliases=(
            "transformers.modeling_utils.AttentionInterface._global_mapping.flash_attention_3",
        ),
        replacement="turbo_physai.optimizations.models.groot.attention.flash_attn_dynamo_disable_wrapper",
    ),
)

DISABLE_KV_CACHE = group(
    "groot.qwen.disable_kv_cache",
    wrap(
        target=f"{_QWEN}.Qwen3VLTextModel.forward",
        replacement="turbo_physai.optimizations.models.groot.kv_cache.disable_kv_cache_wrapper",
    ),
)

_COMPILE = "turbo_physai.optimizations.models.groot.compile.compile_method_wrapper"
_VL_DROPOUT = "turbo_physai.optimizations.models.groot.vl_padding_dropout"

COMPILE_VISION = group(
    "groot.compile.qwen_vision_block",
    wrap(
        target=f"{_QWEN}.Qwen3VLVisionBlock.forward",
        replacement=_COMPILE,
    ),
    depends_on=("groot.qwen.flash_attn_dynamo_disable",),
)

COMPILE_DIT = group(
    "groot.compile.dit",
    *(
        wrap(
            target=f"gr00t.model.modules.dit.{name}.forward",
            replacement=_COMPILE,
        )
        for name in ("DiT", "AlternateVLDiT")
    ),
)

COMPILE_VL = group(
    "groot.compile.vl",
    wrap(
        target="gr00t.model.modules.dit.SelfAttentionTransformer.forward",
        replacement=_COMPILE,
    ),
    wrap(
        target=f"{_VL_DROPOUT}.masked_vl_forward",
        replacement=_COMPILE,
    ),
)

COMPILE_TEXT = group(
    "groot.compile.qwen_text_layer",
    wrap(
        target=f"{_QWEN}.Qwen3VLTextDecoderLayer.forward",
        replacement="turbo_physai.optimizations.models.groot.compile_text.compile_text_wrapper",
    ),
    depends_on=("groot.qwen.flash_attn_dynamo_disable", "groot.qwen.disable_kv_cache"),
)

PIXEL_VALUES_BF16 = group(
    "groot.data.pixel_values_bf16",
    wrap(
        target="gr00t.experiment.trainer.Gr00tTrainer.__init__",
        replacement="turbo_physai.optimizations.models.groot.data.trainer_init_bf16_wrapper",
    ),
)

_FIXED_PADDING = "turbo_physai.optimizations.models.groot.fixed_padding"

FIXED_PADDING = group(
    "groot.data.pad_156",
    wrap(
        target="gr00t.experiment.trainer.Gr00tTrainer.training_step",
        replacement=f"{_FIXED_PADDING}.training_padding_wrapper",
    ),
    wrap(
        target="gr00t.experiment.trainer.Gr00tTrainer._prepare_inputs",
        replacement=f"{_FIXED_PADDING}.fixed_padding_wrapper",
    ),
    wrap(
        target="gr00t.model.gr00t_n1d7.gr00t_n1d7.Gr00tN1d7ActionHead.process_backbone_output",
        replacement=f"{_VL_DROPOUT}.action_head_vl_dropout_wrapper",
    ),
    wrap(
        target="gr00t.model.modules.dit.SelfAttentionTransformer.__init__",
        replacement=f"{_VL_DROPOUT}.vl_dropout_init_wrapper",
    ),
)

APEX_FUSED_ADAM = group(
    "groot.optim.apex_fused_adam",
    wrap(
        target="gr00t.experiment.trainer.Gr00tTrainer.create_optimizer",
        replacement="turbo_physai.optimizations.models.groot.optim.create_optimizer_apex_wrapper",
    ),
    compatibility_check="turbo_physai.optimizations.models.groot.optim.apex_available_check",
)

ZERO2_OVERLAP = group(
    "groot.zero2.grad_reduce_overlap",
    *(
        wrap(
            target=f"deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.{method}",
            replacement=f"turbo_physai.optimizations.models.groot.zero2_overlap.{wrapper}",
        )
        for method, wrapper in (
            ("__init__", "init_wrapper"),
            ("reduce_independent_p_g_buckets_and_remove_grads", "reduce_wrapper"),
            ("setup_buckets", "setup_wrapper"),
            ("average_tensor", "average_wrapper"),
        )
    ),
    compatibility_check="turbo_physai.optimizations.models.groot.zero2_overlap.vendor_helpers_check",
)

ZERO2_OWNER_REDUCE = group(
    "groot.zero2.owner_reduce",
    wrap(
        target="deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.allreduce_and_scatter",
        replacement="turbo_physai.optimizations.models.groot.zero2_owner_reduce.owner_reduce_wrapper",
    ),
    depends_on=("groot.zero2.grad_reduce_overlap",),
    compatibility_check="turbo_physai.optimizations.models.groot.zero2_owner_reduce.owner_reduce_compatibility",
)

ZERO2_ALLGATHER = group(
    "groot.zero2.allgather_pipeline",
    wrap(
        target="transformers.trainer.Trainer.create_scheduler",
        replacement="turbo_physai.optimizations.models.groot.zero2_allgather.scheduler_wrapper",
    ),
    wrap(
        target="gr00t.experiment.trainer.Gr00tTrainer.train",
        replacement="turbo_physai.optimizations.models.groot.zero2_allgather.train_wrapper",
    ),
    wrap(
        target="deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.step",
        replacement="turbo_physai.optimizations.models.groot.zero2_allgather.step_wrapper",
    ),
    wrap(
        target="deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.state_dict",
        replacement="turbo_physai.optimizations.models.groot.zero2_allgather.state_dict_wrapper",
    ),
    wrap(
        target="deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.load_state_dict",
        replacement="turbo_physai.optimizations.models.groot.zero2_allgather.load_state_dict_wrapper",
    ),
    compatibility_check="turbo_physai.optimizations.models.groot.zero2_allgather.allgather_compatibility",
)
