# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""lingbot_vla_v2 optimization declarations.

Importing this module registers the Groups with TurboPhysAI; the generated
OptimizationConfig loads it through ``optimization_modules``.  Declarations are
strings only -- nothing here imports lingbotvla or resolves runtime objects.

The package directory must stay a valid Python identifier (``lingbot_vla_v2``):
``optimization_modules`` and every ``replacement`` path go through ``importlib``.
"""

from __future__ import annotations

from ....engine.definitions import group, replace, wrap

# --- Profiling -----------------------------------------------------------------
# 补上没定义的方法，防止抓 trace 报错。
PROFILER = group(
    "lingbot-vla-v2.profiling",
    replace(
        target="lingbotvla.utils.helper.create_profiler",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.profiling.create_profiler"
        ),
    ),
)

# --- hf_dataset加载优化 -----------------------------------------------------
FAST_HF_QUERY = group(
    "lingbot-vla-v2.fast-hf-query",
    replace(
        target="lingbotvla.data.vla_data.base_dataset.LeRobotDataset._query_hf_dataset",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.fast_hf_query.query_hf_dataset"
        ),
    ),
    wrap(
        target="lingbotvla.data.vla_data.base_dataset.LeRobotDataset.__init__",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.fast_hf_query.wrap_dataset_init"
        ),
    ),
)

# --- Muon 优化器：elementwise 融合 + 小矩阵打包 NS + 免 cat gather ----------
MUON_FUSED = group(
    "lingbot-vla-v2.muon-fused",
    wrap(
        target="lingbotvla.optim.muon.DistributedMuon",
        aliases=("lingbotvla.optim.optimizer.DistributedMuon",),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.muon.configure_optimizer"
        ),
    ),
    replace(
        target="lingbotvla.optim.muon.batched_newton_schulz",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.muon.batched_newton_schulz"
        ),
    ),
)

# --- flex attention: HCU 上的 tile 调优 ---------------------------------
FLEX_KERNEL_OPTIONS = group(
    "lingbot-vla-v2.flex-kernel-options",
    wrap(
        target="lingbotvla.models.vla.lingbot_vla.flex_attention.flex_attention",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.flex_kernel_options.flex_attention_kernel_options"
        ),
    ),
)

# --- 3d 卷积改走 channels_last_3d --------------------------------------------
VISION_PATCH_EMBED = group(
    "lingbot-vla-v2.conv3d-channel-last",
    replace(
        target=(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.conv3d_channel_last.forward"
        ),
    ),
)

# --- Qwen3vl Vision tower: 输入图像尺寸不变，索引表可以缓存 ----------------------
QWEN3VL_CACHE_GRID_THW = group(
    "lingbot-vla-v2.qwen3vl-cache-grid-thw",
    replace(
        target=(
            "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla.Qwen3VLVisionModel.fast_pos_embed_interpolate"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_cache_gridTHW.fast_pos_embed_interpolate"
        ),
    ),
)

# --- Qwen3-VL vision: 逐层编译 -----------------------------------
QWEN3VL_COMPILE = group(
    "lingbot-vla-v2.qwen3vl-compile",
    wrap(
        target=(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2.FlowMatchingV2.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_vision.wrap_init"
        ),
    ),
    wrap(
        target="lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla.Qwen3VLVisionModel.__init__",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_vision.install_on_block_init"
        ),
    ),
    replace(
        target=(
            "lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla.forward_without_grid_thw"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_vision.compile_vision_blocks"
        ),
    ),
)

# --- Qwen3-VL text: 逐层编译 -------------------------------------
QWEN3VL_COMPILE_TEXT = group(
    "lingbot-vla-v2.qwen3vl-compile-text",
    wrap(
        target=(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2.QwenvlWithExpertV2Model.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_text.install_on_init"
        ),
    ),
    replace(
        target=(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2.QwenvlWithExpertV2Model.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_compile_text.compile_text_layers"
        ),
    ),
)

# --- Qwen2 action expert: 逐层编译 -------------------------------
QWEN2_COMPILE_EXPERT = group(
    "lingbot-vla-v2.qwen2-compile-expert",
    wrap(
        target=(
            "lingbotvla.models.vla.lingbot_vla.qwen2_action_expert.Qwen2ForCausalLM.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen2_compile_expert.install_on_init"
        ),
    ),
    wrap(
        target=(
            "lingbotvla.models.vla.lingbot_vla.qwen2_action_expert.Qwen2DecoderLayer.__call__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen2_compile_expert.install_call_guard"
        ),
    ),
)

# --- Qwen3VLModel.get_rope_index 放在CPU算避免频繁同步 --------------------------------------------
QWEN3VL_ROPE_INDEX_CPU = group(
    "lingbot-vla-v2.qwen3vl-rope-index-cpu",
    replace(
        target=(
            "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel.get_rope_index"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.qwen3vl_rope_index_cpu.get_rope_index"
        ),
    ),
)

# --- MoGe: 跳过没被使用的norm head --------------
MOGE_DEPTH_ONLY = group(
    "lingbot-vla-v2.moge-depth-only",
    wrap(
        target="lingbotvla.models.vla.vision_models.module_utils.v2.infer",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.moge_depth_only.wrap_infer"
        ),
    ),
    depends_on=("lingbot-vla-v2.compile-teacher",),
)

# --- 编译teacher网络 -------------------------
TEACHER_COMPILE = group(
    "lingbot-vla-v2.compile-teacher",
    wrap(
        target="lingbotvla.models.vla.vision_models.module_utils.v2.forward",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.compile_teacher.compiled_forward"
        ),
    ),
    wrap(
        target=(
            "lingbotvla.models.vla.vision_models.module_utils.v2_morgbd.forward_feat"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.compile_teacher.compiled_forward"
        ),
    ),
    wrap(
        target=(
            "lingbotvla.models.vla.vision_models.dino_video.lumos_dinov3.models.inference.CausalEvalAdapter.get_intermediate_layers"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.compile_teacher.compiled_forward"
        ),
    ),
    replace(
        target="lingbotvla.models.vla.vision_models.module_utils.get_depth_target",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.compile_teacher.fast_get_depth_target"
        ),
    ),
)

# --- DINOv2 使用 FA2 -------------------------------
TEACHER_FA2 = group(
    "lingbot-vla-v2.teacher-fa2",
    wrap(
        target="lingbotvla.models.vla.vision_models.module_utils.build_depth_model",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.teacher_fa2.wrap_build_depth_model"
        ),
    ),
)

# --- Video teacher 使用 FA2 -------------------
VIDEO_TEACHER_FA2 = group(
    "lingbot-vla-v2.video-teacher-fa2",
    wrap(
        target=(
            "lingbotvla.models.vla.vision_models.dino_video.teacher.DinoVideoTeacher.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.video_fa2.wrap_teacher_init"
        ),
    ),
    replace(
        target=(
            "lingbotvla.models.vla.vision_models.dino_video.lumos_dinov3.layers.sdpa_attention_packed.sdpa_attention_block_causal"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.video_fa2.flash_attention_block_causal"
        ),
    ),
)

# --- 36 层共用同一个 position_ids：cos/sin 缓存一次；旋转使用 compile 融合。 -----------------------------
ROPE_FAST = group(
    "lingbot-vla-v2.rope-fast",
    replace(
        target=(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2."
            "QwenvlWithExpertV2Model.apply_mrope"
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.rope_fast.apply_mrope"
        ),
    ),
)

# --- flex attention优化 ---------
FLEX_HEAD_SHARED_MASK = group(
    "lingbot-vla-v2.flex-head-shared-mask",
    wrap(
        target="lingbotvla.models.vla.lingbot_vla.flex_attention.build_block_mask",
        aliases=(
            "lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2.build_block_mask",
        ),
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2.flex_head_shared_mask.wrap_build_block_mask"
        ),
    ),
)

# --- flex attention: fp32 dot 开启 tf32 ------------------------------------------
FLEX_ATTN_TF32 = group(
    "lingbot-vla-v2.flex-attn-tf32",
    wrap(
        target="torch._inductor.kernel.flex.flex_attention.get_float32_precision",
        replacement=(
            "turbo_physai.optimizations.models.lingbot_vla_v2."
            "flex_attn_tf32.wrap_get_float32_precision"
        ),
    ),
)
