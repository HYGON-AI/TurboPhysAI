# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Cosmos3 Generator, Action Policy and Reasoner optimization catalog.

Targets the Cosmos framework at upstream baseline
9726697a83315540c6885baefd2fe353d9c74920; implementations are ported from the
HCU-adapted commit 291c4e231809273d850910f07e9cdb252e9bd244.

Alias notes (verified against the baseline call graph):
- ``flash2_attention`` is re-exported by ``model.attention.flash2.__init__``
  and from-imported once by ``model.attention.frontend`` — both are aliased.
- ``dispatch_attention`` is from-imported by ``mot.unified_mot``,
  ``mot.parallelize_unified_mot`` and ``mot.inference_text_kv_memory``.
- ``FLASH2_SUPPORTED`` is computed at import time in ``flash2.__init__`` and
  from-imported by ``flash2.checks``; both module attributes are replaced with
  the widened HIP evaluation so the runtime flag matches the check function.
- ``cross_entropy_loss`` is from-imported by ``vlm_model``.
- ``init_sequence_pack`` has no from-import consumers at baseline.
"""

from ....engine.definitions import group, replace, replace_import, wrap
from ....bootstrap._toml import ensure_tomllib

_C3 = "turbo_physai.optimizations.models.cosmos3"

# --------------------------------------------------------------------------
# Python compatibility precedes model imports, including direct apply/check.
# This is an idempotent fallback, not an Engine import replacement: startup
# hooks and Python 3.11+ may already have loaded a valid tomllib.
ensure_tomllib()

COMPILE_OPTIONS = group(
    'cosmos3.compile_options',
    replace('cosmos_framework.utils.misc.set_torch_compile_options', f'{_C3}.compile_options.set_torch_compile_options'),
)

DEVICE_MONITOR = group(
    'cosmos3.device_monitor',
    replace('cosmos_framework.callbacks.device_monitor.DeviceMonitor.on_train_start', f'{_C3}.device_monitor.on_train_start'),
    wrap('cosmos_framework.callbacks.device_monitor.DeviceMonitor.every_n_impl', f'{_C3}.device_monitor.guard_every_n'),
)

DTENSOR_NORM = group(
    'cosmos3.dtensor_norm',
    replace('cosmos_framework.callbacks.grad_clip._clip_grad', f'{_C3}.grad_clip._clip_grad'),
)

DATAFLOW_CONTEXT = group(
    'cosmos3.dataflow_context',
    replace('cosmos_framework.data.generator.dataflow.loader.CosmosDataLoader.__init__', f'{_C3}.dataflow.initialize'),
)

# --------------------------------------------------------------------------
# Attention: HIP FlashAttention v2 (varlen direct wrappers), backend order,
# arch tag, and the import-time support flag.
FLASH2_HIP = group(
    'cosmos3.flash2_hip',
    # flash2.__init__ selected functions vs stubs at import time from the
    # CUDA-capped FLASH2_SUPPORTED flag.  The package re-export and every
    # from-import copy are replaced with the ported HIP implementation, whose
    # module recomputes the flag against the widened HIP version range, so the
    # stub selection made at import time no longer reaches any caller.
    replace(
        'cosmos_framework.model.attention.flash2.flash2_attention',
        f'{_C3}.attention.flash2_hip.flash2_attention',
        aliases=(
            'cosmos_framework.model.attention.frontend.flash2_attention',
            'cosmos_framework.model.attention.frontend.BACKEND_MAP.flash2',
        ),
    ),
    # The defining module keeps a separate binding when FLASH2_SUPPORTED was
    # False at import (stubs exported instead); replace it independently.
    replace(
        'cosmos_framework.model.attention.flash2.functions.flash2_attention',
        f'{_C3}.attention.flash2_hip.flash2_attention',
    ),
    replace(
        'cosmos_framework.model.attention.flash2.checks.flash2_attention_check',
        f'{_C3}.attention.flash2_hip.flash2_attention_check',
        aliases=('cosmos_framework.model.attention.backends.BACKEND_CHECK_MAP.flash2',),
    ),
    replace(
        'cosmos_framework.model.attention.backends.get_backend_list',
        f'{_C3}.attention.flash2_hip.get_backend_list',
    ),
    replace(
        'cosmos_framework.model.attention.utils.get_arch_tag',
        f'{_C3}.attention.flash2_hip.get_arch_tag',
        aliases=(
            'cosmos_framework.model.attention.backends.get_arch_tag',
            'cosmos_framework.model.attention.flash2.checks.get_arch_tag',
            'cosmos_framework.model.attention.cudnn.checks.get_arch_tag',
            'cosmos_framework.model.attention.flash3.checks.get_arch_tag',
            'cosmos_framework.model.attention.natten.checks.get_arch_tag',
        ),
    ),
)

HF_ATTENTION = group(
    'cosmos3.hf_attention',
    # hf_model registers this function object with transformers'
    # AttentionInterface inside HFModel.__init__ via a local import, so
    # replacing the module attribute before model construction is sufficient.
    replace(
        'cosmos_framework.utils.generator.hf_attention_cosmos.hf_attention_cosmos',
        f'{_C3}.attention.hf_attention.hf_attention_cosmos',
    ),
)

# --------------------------------------------------------------------------
# Sequence packing + MoT attention (paired: _fa_max_len reads the fa_* keys
# init_sequence_pack writes).
SEQUENCE_PACKING = group(
    'cosmos3.sequence_packing',
    replace(
        'cosmos_framework.data.generator.sequence_packing.runtime.init_sequence_pack',
        f'{_C3}.sequence_packing.init_sequence_pack',
    ),
    replace(
        'cosmos_framework.model.generator.mot.attention.two_way_attention',
        f'{_C3}.mot_attention.two_way_attention',
    ),
    replace(
        'cosmos_framework.model.generator.mot.attention.three_way_attention',
        f'{_C3}.mot_attention.three_way_attention',
    ),
    replace(
        'cosmos_framework.model.generator.mot.attention.multi_control_two_way_attention',
        f'{_C3}.mot_attention.multi_control_two_way_attention',
    ),
    replace(
        'cosmos_framework.model.generator.mot.attention.dispatch_attention',
        f'{_C3}.mot_attention.dispatch_attention',
        aliases=(
            'cosmos_framework.model.generator.mot.unified_mot.dispatch_attention',
            'cosmos_framework.model.generator.mot.parallelize_unified_mot.dispatch_attention',
            'cosmos_framework.model.generator.mot.inference_text_kv_memory.dispatch_attention',
        ),
    ),
)

# --------------------------------------------------------------------------
# FSDP: ordered collectives + fsdp_layers_per_group grouping for the VLM path,
# ordered loss reductions, and the DTensor-norm/loss stream sharing.
FSDP_VLM = group(
    'cosmos3.fsdp_vlm',
    replace(
        'cosmos_framework.model.generator.parallelize_vlm.apply_fsdp',
        f'{_C3}.parallelize_vlm.apply_fsdp',
    ),
    replace(
        'cosmos_framework.model.generator.algorithm.loss.cross_entropy.cross_entropy_loss',
        f'{_C3}.losses.cross_entropy_loss',
        aliases=('cosmos_framework.model.generator.vlm_model.cross_entropy_loss',),
    ),
    replace(
        'cosmos_framework.model.generator.vlm_model.VLMModel.training_step',
        f'{_C3}.losses.vlm_training_step',
    ),
)

# --------------------------------------------------------------------------
# Generator model compute: channels-last conv weights before FSDP, CPU-side
# timestep handoff, distributed-optional parallel dims.
GENERATOR_MODEL = group(
    'cosmos3.generator_model',
    replace('cosmos_framework.model.generator.omni_mot_model.OmniMoTModel.build_net', f'{_C3}.omni_model.build_net'),
    replace('cosmos_framework.model.generator.omni_mot_model.OmniMoTModel.training_step', f'{_C3}.omni_model.training_step'),
    replace('cosmos_framework.model.generator.omni_mot_model.OmniMoTModel.set_up_parallelism', f'{_C3}.omni_model.set_up_parallelism'),
    # Trainer moves batches with misc.to(...); keep image_size metadata on CPU
    # for HIP resolution lookups.  The wrapper dispatches by payload shape, so
    # non-batch misc.to callers keep the original behavior.
    wrap('cosmos_framework.utils.misc.to', f'{_C3}.batch_transfer.wrap_trainer_batch_transfer'),
)

# --------------------------------------------------------------------------
# Wan VAE tokenizer: channels-last layout end-to-end + optional hipDNN fused
# causal Conv3d (default OFF, env-gated exactly like the source).
VAE_LAYOUT = group(
    'cosmos3.vae_layout',
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16._contiguous_clone', f'{_C3}.vae.layout._contiguous_clone'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16._update_cache_and_apply', f'{_C3}.vae.layout._update_cache_and_apply'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.CausalConv3d.forward', f'{_C3}.vae.layout.causal_conv3d_forward'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Resample.forward', f'{_C3}.vae.layout.resample_forward'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.AttentionBlock.forward', f'{_C3}.vae.layout.attention_block_forward'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.AvgDown3D.forward', f'{_C3}.vae.layout.avg_down3d_forward'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.DupUp3D.forward', f'{_C3}.vae.layout.dup_up3d_forward'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.WanVAE_._encode_chunk_impl', f'{_C3}.vae.wan_vae.wan_vae_encode_chunk_impl'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.WanVAE_.encode', f'{_C3}.vae.wan_vae.wan_vae_inner_encode'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.WanVAE.encode', f'{_C3}.vae.wan_vae.wan_vae_encode'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.WanVAE.decode', f'{_C3}.vae.wan_vae.wan_vae_decode'),
    wrap('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.WanVAE.__init__', f'{_C3}.vae.wan_vae.wan_vae_post_init'),
    wrap('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface.__init__', f'{_C3}.vae.wan_vae.interface_chunk_frames'),
)

VAE_COMPILE = group(
    'cosmos3.vae_compile',
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16._collect_warmup_shapes', f'{_C3}.vae.wan_vae._collect_warmup_shapes'),
    replace('cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface.compile_encode', f'{_C3}.vae.wan_vae.interface_compile_encode'),
    depends_on=('cosmos3.vae_layout',),
)

# --------------------------------------------------------------------------
# Qwen3-VL encoder used by the Generator (VLM text/vision tower).
QWEN_ENCODER = group(
    'cosmos3.qwen_encoder',
    wrap('cosmos_framework.model.generator.hf_model.HFModel.__init__', f'{_C3}.qwen.qwen_local_registration'),
    replace('cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl.Qwen3VLVisionPatchEmbed.forward', f'{_C3}.qwen.vision_patch_embed_forward'),
    replace('cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe.Qwen3VLMoeVisionPatchEmbed.forward', f'{_C3}.qwen.moe_vision_patch_embed_forward'),
    replace('cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl.Qwen3VLVisionAttention.forward', f'{_C3}.qwen.vision_attention_forward'),
    replace('cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl.Qwen3VLVisionBlock.forward', f'{_C3}.qwen.vision_block_forward'),
    wrap('cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl.Qwen3VLVisionModel.__init__', f'{_C3}.qwen.qwen_vision_model_post_init'),
    replace('cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl.Qwen3VLVisionModel.forward', f'{_C3}.qwen.vision_model_forward'),
    replace(
        'cosmos_framework.model.generator.reasoner.qwen3_vl.utils.get_rope_index',
        f'{_C3}.qwen.get_rope_index',
        aliases=('cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl._get_rope_index',),
    ),
)

# --------------------------------------------------------------------------
# Checkpoint: DCP metadata Gloo group, serial copy-ahead staging, async
# fallback, RNG-tolerant load with EMA warm-start reseed.
CHECKPOINT_DCP = group(
    'cosmos3.checkpoint_dcp',
    wrap('cosmos_framework.checkpoint.dcp.DistributedCheckpointer.__init__', f'{_C3}.checkpoint_dcp.checkpointer_post_init'),
    replace('cosmos_framework.checkpoint.dcp.DistributedCheckpointer.get_storage_writer', f'{_C3}.checkpoint_dcp.get_storage_writer'),
    replace('cosmos_framework.checkpoint.dcp.DistributedCheckpointer._checkpoint_async_with_pinned_memory', f'{_C3}.checkpoint_dcp._checkpoint_async_with_pinned_memory'),
    replace('cosmos_framework.checkpoint.dcp.DistributedCheckpointer.save_state_dict_worker', f'{_C3}.checkpoint_dcp.save_state_dict_worker'),
    replace('cosmos_framework.checkpoint.dcp.DistributedCheckpointer.load', f'{_C3}.checkpoint_dcp.load'),
)

# --------------------------------------------------------------------------
# Data pipeline: worker-kwargs hygiene, audio-stream probing.
DATA_PIPELINE = group(
    'cosmos3.data_pipeline',
    replace('cosmos_framework.data.generator.joint_dataloader.RankPartitionedDataLoader.__init__', f'{_C3}.joint_dataloader.rank_partitioned_init'),
    replace('cosmos_framework.data.generator.augmentors.video_parsing.VideoParsing._extract_audio_chunk', f'{_C3}.video_audio._extract_audio_chunk'),
)

# --------------------------------------------------------------------------
# Runtime compatibility: NVML-optional distributed init.
RUNTIME_INIT = group(
    'cosmos3.runtime_init',
    replace('cosmos_framework.utils.distributed.init', f'{_C3}.distributed_init.init'),
)


# All Cosmos3 training scenarios share the default optimization config.
# The Cosmos TOML recipe selects the training workload.
VIDEO_DECODE = group(
    'cosmos3.video_decode',
    replace_import('cosmos_framework.utils.video_decode', f'{_C3}.video_decode'),
)
TRAINING_WORKERS = group(
    'cosmos3.training_workers',
    wrap('torch.utils.data.DataLoader.__init__', f'{_C3}.training_workers.wrap_dataloader_init'),
)
LOAD_ONLY = group(
    'cosmos3.load_only',
    replace_import('cosmos_framework.checkpoint.load_only', f'{_C3}.load_only'),
)
_ACTION = 'cosmos_framework.data.generator.action.datasets'
ACTION_POLICY = group(
    'cosmos3.action_policy',
    replace(f'{_ACTION}.cosmos3_action_lerobot._LRUVideoDecoderCache', f'{_C3}.action_lerobot._LRUVideoDecoderCache'),
    replace(f'{_ACTION}.cosmos3_action_lerobot._patch_decoder_cache', f'{_C3}.action_lerobot._patch_decoder_cache'),
    replace(f'{_ACTION}.cosmos3_action_lerobot.BaseActionLeRobotDataset.__init__', f'{_C3}.action_lerobot.__init__'),
    replace(f'{_ACTION}.cosmos3_action_lerobot.BaseActionLeRobotDataset._register_sources', f'{_C3}.action_lerobot._register_sources'),
    replace(f'{_ACTION}.cosmos3_action_lerobot.BaseActionLeRobotDataset._get_dataset', f'{_C3}.action_lerobot._get_dataset'),
    replace(f'{_ACTION}.droid_lerobot_dataset.DROIDLeRobotDataset.__init__', f'{_C3}.action_droid.__init__'),
    replace(f'{_ACTION}.action_sft_dataset.get_action_droid_sft_dataset', f'{_C3}.action_sft.get_action_droid_sft_dataset'),
    depends_on=('cosmos3.training_workers', 'cosmos3.data_pipeline'),
)
ACTION_RECIPE = group(
    'cosmos3.action_recipe',
    replace_import('cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano', f'{_C3}.recipes.action_policy_droid_nano'),
)
REASONER_VIDEO = group(
    'cosmos3.reasoner_video',
    replace('cosmos_framework.data.generator.reasoner.video_decoder_qwen._video_decoder_qwen_func', f'{_C3}.reasoner_video._video_decoder_qwen_func'),
    replace('cosmos_framework.data.generator.augmentors.interleaved_video_parsing._create_video_decoder', f'{_C3}.interleaved_video._create_video_decoder'),
    replace('cosmos_framework.data.generator.augmentors.interleaved_video_parsing.VideoTransferAlignedFullFramesParsing._probe_video_len', f'{_C3}.interleaved_video._probe_video_len'),
    depends_on=('cosmos3.training_workers', 'cosmos3.dataflow_context'),
)
REASONER_RECIPES = group(
    'cosmos3.reasoner_recipes',
    replace_import('cosmos_framework.configs.base.reasoner.experiment.videophy2_dataflow_roles', f'{_C3}.recipes.videophy2_dataflow_roles'),
    replace_import('cosmos_framework.configs.base.reasoner.experiment.videophy2_sft_nano', f'{_C3}.recipes.videophy2_sft_nano'),
    replace_import('cosmos_framework.configs.base.reasoner.experiment.videophy2_sft_edge', f'{_C3}.recipes.videophy2_sft_edge'),
)
