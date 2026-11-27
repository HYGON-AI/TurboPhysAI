# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ....engine.definitions import group, replace, wrap

FLEX_ATTN = group(
    "lingbot-vla.flex-attn",
    replace(
        target="lingbotvla.models.vla.pi0.modeling_lingbot_vla.flex_attention_forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.flex_attn.flex_attention_forward"
        ),
    ),
    wrap(
        target="torch._inductor.kernel.flex.flex_attention.get_float32_precision",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "flex_attn.wrap_get_float32_precision"
        ),
    ),
)

GRADIENT_CHECKPOINTING = group(
    "lingbot-vla.gradient-checkpointing",
    wrap(
        target="lingbotvla.distributed.torch_parallelize.build_parallelize_model",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "gradient_checkpointing.wrap_enable_gc"
        ),
    ),
    replace(
        target="lingbotvla.models.vla.pi0.modeling_lingbot_vla.AdaRMSNorm.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "gradient_checkpointing.ada_rms_norm_forward"
        ),
    ),
    replace(
        target="lingbotvla.models.vla.pi0.modeling_lingbot_vla.QwenvlWithExpertModel.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.gradient_checkpointing.forward"
        ),
    ),
    wrap(
        target="lingbotvla.models.vla.pi0.modeling_lingbot_vla.QwenvlWithExpertModel.__init__",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "gradient_checkpointing.wrap_llm_gc"
        ),
    ),
)

FAST_HF_QUERY = group(
    "lingbot-vla.fast-hf-query",
    replace(
        target="lingbotvla.data.vla_data.base_dataset.LeRobotDataset._query_hf_dataset",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "fast_hf_query.query_hf_dataset"
        ),
    ),
)

GC_FREEZE = group(
    "lingbot-vla.gc-freeze",
    wrap(
        target=(
            "lingbotvla.distributed.offloading.build_activation_offloading_context"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.gc_freeze."
            "wrap_build_activation_offloading_context"
        ),
    ),
    wrap(
        target="torch.cuda.synchronize",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.gc_freeze."
            "wrap_synchronize"
        ),
    ),
)

MOGE_DEPTH_ONLY = group(
    "lingbot-vla.moge-depth-only",
    wrap(
        target="moge.model.v2.MoGeModel.infer",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.moge_depth_only.wrap_infer"
        ),
    ),
    replace(
        target="lingbotvla.models.vla.vision_models.module_utils.get_depth_target",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.moge_depth_only.fast_get_depth_target"
        ),
    ),
)

PROFILING = group(
    "lingbot-vla.profiling",
    replace(
        target="lingbotvla.utils.helper.create_profiler",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.profiling.create_profiler"
        ),
    ),
)

FUSED_ADAMW = group(
    "lingbot-vla.fused-adamw",
    wrap(
        target="lingbotvla.optim.build_optimizer",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.fused_adamw.wrap_build_optimizer"
        ),
    ),
    wrap(
        target="torch.nn.utils.clip_grad_norm_",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.fused_adamw.wrap_clip_grad_norm"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.OptimizerState.state_dict",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.fused_adamw."
            "wrap_optimizer_state_dict"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.OptimizerState.load_state_dict",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.fused_adamw."
            "wrap_load_optimizer_state_dict"
        ),
    ),
)

PER_LAYER_COMPILE = group(
    "lingbot-vla.per-layer-compile",
    wrap(
        target="torch.compile",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.compile_vision.wrap_torch_compile"
        ),
    ),
)

TEACHER_COMPILE = group(
    "lingbot-vla.teacher-compile",
    wrap(
        target="mdm.model.v2.MDMModel.infer_feat",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "compile_teacher.wrap_infer_feat"
        ),
    ),
)

POS_EMBED_CACHE = group(
    "lingbot-vla.pos-embed-cache",
    wrap(
        target=(
            "mdm.model.dinov2_rgbd.models.vision_transformer."
            "DinoVisionTransformer.interpolate_pos_encoding_without_cls"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "pos_embed_cache.wrap_interpolate_pos_encoding_without_cls"
        ),
    ),
)

PARAM_DEVICE = group(
    "lingbot-vla.param-device",
    wrap(
        target=(
            "lingbotvla.models.vla.pi0.modeling_lingbot_vla."
            "FlowMatching.init_depth_heads"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "param_device.wrap_init_depth_heads"
        ),
    ),
)

GRID_THW_CPU = group(
    "lingbot-vla.grid-thw-cpu",
    wrap(
        target=(
            "lingbotvla.models.vla.pi0.modeling_lingbot_vla."
            "QwenvlWithExpertModel.embed_image"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "grid_thw_cpu.wrap_embed_image"
        ),
    ),
    wrap(
        target=(
            "lingbotvla.models.vla.pi0.qwenvl_in_vla."
            "Qwen2_5_VisionTransformerPretrainedModel.preprcess_grid_thw"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "grid_thw_cpu.wrap_preprocess_grid_thw"
        ),
    ),
)

VISION_FA2 = group(
    "lingbot-vla.vision-fa2-fixed",
    replace(
        target="lingbotvla.models.vla.pi0.qwenvl_in_vla.Qwen2_5_VLVisionFlashAttention2.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.vision_fa2.forward"
        ),
    ),
)

VISION_ROPE_FUSED = group(
    "lingbot-vla.vision-rope-fused",
    replace(
        target="lingbotvla.models.vla.pi0.qwenvl_in_vla.apply_rotary_pos_emb_flashatt",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.vision_rope.apply_rotary_pos_emb_flashatt"
        ),
    ),
)

ROPE_CACHE = group(
    "lingbot-vla.rope-cache",
    replace(
        target="lingbotvla.models.vla.pi0.utils.apply_rope",
        aliases=(
            "lingbotvla.models.vla.pi0.modeling_lingbot_vla.apply_rope",
            "lingbotvla.models.vla.pi0.modeling_pi0.apply_rope",
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.rope_cache.apply_rope"
        ),
    ),
)

MOGE_FOCAL_GPU = group(
    "lingbot-vla.moge-focal-gpu",
    replace(
        target="moge.model.v2.recover_focal_shift",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.moge_focal_gpu.recover_focal_shift"
        ),
    ),
)

QKV_FUSED = group(
    "lingbot-vla.qkv-fused",
    wrap(
        target="lingbotvla.models.loader.CustomizedModelingLoader.load_model",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused.wrap_load_model"
        ),
    ),
    replace(
        target="lingbotvla.models.vla.pi0.modeling_lingbot_vla.Qwen2DecoderLayer.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused.layer_forward"
        ),
    ),
    replace(
        target="lingbotvla.models.vla.pi0.qwenvl_in_vla.Qwen2_5_VLDecoderLayer.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused.vl_layer_forward"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.ModelState.state_dict",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused."
            "wrap_model_state_dict"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.ModelState.load_state_dict",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused."
            "wrap_load_model_state_dict"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.get_optimizer_state_dict",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused."
            "wrap_get_optimizer_state_dict"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.set_optimizer_state_dict",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused."
            "wrap_set_optimizer_state_dict"
        ),
    ),
    wrap(
        target="lingbotvla.checkpoint.checkpointer.build_checkpointer",
        aliases=("lingbotvla.checkpoint.build_checkpointer",),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.qkv_fused."
            "wrap_build_checkpointer"
        ),
    ),
)

PATCH_EMBED_MATMUL = group(
    "lingbot-vla.patch-embed-matmul",
    replace(
        target="lingbotvla.models.vla.pi0.qwenvl_in_vla.Qwen2_5_VisionPatchEmbed.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla.patch_embed_matmul.forward"
        ),
    ),
)

MOGE_RESAMPLER = group(
    "lingbot-vla.moge-resampler",
    wrap(
        target="moge.model.modules.Resampler.__init__",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "moge_resampler.wrap_resampler_init"
        ),
    ),
    wrap(
        target="mdm.model.modules_decoder.Resampler.__init__",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla."
            "moge_resampler.wrap_resampler_init"
        ),
    ),
)
