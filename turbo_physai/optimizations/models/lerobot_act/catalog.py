# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python catalog declarations for LeRobot ACT HCU optimizations."""

from __future__ import annotations

from ....engine.definitions import group, replace, wrap


BASE = "turbo_physai.optimizations.models.lerobot_act"
ACT = "lerobot.policies.act.modeling_act"


CHANNELS_LAST_POOL = group(
    "lerobot_act.channels_last_pool",
    wrap(
        target=f"{ACT}.ACT.__init__",
        replacement=f"{BASE}.model_optimizations.channels_last_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.conv.Conv2d.forward",
        replacement=f"{BASE}.model_optimizations.channels_last_input_wrapper",
    ),
    wrap(
        target="lerobot.optim.optimizers._save_single_optimizer_state",
        replacement=f"{BASE}.model_optimizations.contiguous_optimizer_state_wrapper",
    ),
    wrap(
        target="torch.nn.modules.pooling.MaxPool2d.forward",
        replacement=f"{BASE}.pool_backward_gather.max_pool_backward_gather_wrapper",
    ),
)

BACKBONE_COMPILE = group(
    "lerobot_act.backbone_compile",
    wrap(
        target="torchvision.models._utils.IntermediateLayerGetter.forward",
        replacement=f"{BASE}.model_optimizations.backbone_compile_wrapper",
    ),
)

NO_ATTN_AVERAGE = group(
    "lerobot_act.no_attn_average",
    wrap(
        target="torch.nn.modules.activation.MultiheadAttention.forward",
        replacement=f"{BASE}.model_optimizations.attention_no_average_wrapper",
    ),
)

FFN_COMPILE = group(
    "lerobot_act.ffn_compile",
    wrap(
        target=f"{ACT}.ACTEncoderLayer.forward",
        replacement=f"{BASE}.model_optimizations.encoder_ffn_compile_wrapper",
    ),
    wrap(
        target=f"{ACT}.ACTDecoderLayer.forward",
        replacement=f"{BASE}.model_optimizations.decoder_ffn_compile_wrapper",
    ),
)

ATTENTION_SOFTMAX_DROPOUT = group(
    "lerobot_act.attention_softmax_dropout",
    wrap(
        target="torch.nn.functional.multi_head_attention_forward",
        replacement=f"{BASE}.attention_softmax_dropout.attention_softmax_dropout_wrapper",
    ),
)

QK_TF32 = group(
    "lerobot_act.qk_tf32",
    replace(
        target=f"{BASE}.attention_softmax_dropout._unmasked_qk",
        replacement=f"{BASE}.qk_tf32.qk_bmm",
    ),
    depends_on=("lerobot_act.attention_softmax_dropout",),
)

DDP_OPTIONS = group(
    "lerobot_act.ddp_options",
    wrap(
        target="torch.nn.parallel.distributed.DistributedDataParallel.__init__",
        replacement=f"{BASE}.training_ddp_h2d.ddp_options_wrapper",
    ),
)

NONBLOCKING_H2D = group(
    "lerobot_act.nonblocking_h2d",
    wrap(
        target="accelerate.accelerator.Accelerator.__init__",
        replacement=f"{BASE}.training_ddp_h2d.nonblocking_h2d_wrapper",
    ),
)

H2D_PREFETCH_STREAM = group(
    "lerobot_act.h2d_prefetch_stream",
    replace(
        target="accelerate.data_loader.DataLoaderShard.__iter__",
        replacement=f"{BASE}.h2d_prefetch_stream.prefetch_dataloader_iter",
    ),
    depends_on=("lerobot_act.nonblocking_h2d",),
)

DEFERRED_METRICS_APEX_LAYERNORM = group(
    "lerobot_act.deferred_metrics_apex_layernorm",
    wrap(
        target=f"{ACT}.ACTPolicy.forward",
        replacement=f"{BASE}.policy_metrics_layernorm.deferred_metrics_policy_wrapper",
    ),
    wrap(
        target="torch.nn.modules.normalization.LayerNorm.forward",
        replacement=f"{BASE}.policy_metrics_layernorm.apex_layernorm_backward_wrapper",
    ),
)

SAMPLER_VECTOR = group(
    "lerobot_act.sampler_vector",
    wrap(
        target="lerobot.datasets.sampler.EpisodeAwareSampler._iter_epoch",
        replacement=f"{BASE}.sampler_vector.vectorized_sampler_wrapper",
    ),
)

INFINITE_BATCH_SAMPLER = group(
    "lerobot_act.infinite_batch_sampler",
    wrap(
        target="accelerate.data_loader.DataLoaderShard.set_epoch",
        replacement=f"{BASE}.infinite_batch_sampler.preserve_resume_epoch_wrapper",
    ),
    wrap(
        target="accelerate.data_loader.BatchSamplerShard.__iter__",
        replacement=f"{BASE}.infinite_batch_sampler.continuous_batches_wrapper",
    ),
)

VAE_NO_CLONE = group(
    "lerobot_act.vae_no_clone",
    wrap(
        target=f"{ACT}.ACT.forward",
        replacement=f"{BASE}.vae_no_clone.vae_no_clone_wrapper",
    ),
)

APEX_FUSED_ADAMW = group(
    "lerobot_act.apex_fused_adamw",
    wrap(
        target="lerobot.optim.optimizers.AdamWConfig.build",
        replacement=f"{BASE}.apex_fused_adamw.apex_fused_adamw_wrapper",
    ),
)
