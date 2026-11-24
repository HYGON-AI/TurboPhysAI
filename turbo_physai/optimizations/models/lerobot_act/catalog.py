# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
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
        replacement=f"{BASE}.channels_last.channels_last_init_wrapper",
    ),
    wrap(
        target="torch.nn.modules.conv.Conv2d.forward",
        replacement=f"{BASE}.channels_last.channels_last_input_wrapper",
    ),
    wrap(
        target="lerobot.optim.optimizers._save_single_optimizer_state",
        replacement=f"{BASE}.channels_last.contiguous_optimizer_state_wrapper",
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
        replacement=f"{BASE}.backbone_compile.backbone_compile_wrapper",
    ),
)

NO_ATTN_AVERAGE = group(
    "lerobot_act.no_attn_average",
    wrap(
        target="torch.nn.modules.activation.MultiheadAttention.forward",
        replacement=f"{BASE}.no_attn_average.attention_no_average_wrapper",
    ),
)

MODULE_COMPILE = group(
    "lerobot_act.module_compile",
    wrap(
        target=f"{ACT}.ACTEncoder.forward",
        replacement=f"{BASE}.module_compile.wrap_encoder",
    ),
    wrap(
        target=f"{ACT}.ACTDecoder.forward",
        replacement=f"{BASE}.module_compile.wrap_decoder",
    ),
)

COMPILED_ATTENTION_SCORES = group(
    "lerobot_act.compiled_attention_scores",
    wrap(
        target="torch._inductor.async_compile.AsyncCompile.triton",
        replacement=f"{BASE}.compiled_attention_scores.wrap_async_triton",
    ),
    depends_on=("lerobot_act.module_compile",),
    compatibility_check=f"{BASE}.compiled_attention_scores.check_software_stack",
)

QK_TF32 = group(
    "lerobot_act.qk_tf32",
    wrap(
        target="torch.nn.functional.multi_head_attention_forward",
        replacement=f"{BASE}.qk_tf32.attention_qk_tf32_wrapper",
        runtime_condition=f"{BASE}.qk_tf32.supports_attention_qk_tf32",
    ),
)

DDP_OPTIONS = group(
    "lerobot_act.ddp_options",
    wrap(
        target="torch.nn.parallel.distributed.DistributedDataParallel.__init__",
        replacement=f"{BASE}.ddp_options.ddp_options_wrapper",
    ),
)

NONBLOCKING_H2D = group(
    "lerobot_act.nonblocking_h2d",
    wrap(
        target="accelerate.accelerator.Accelerator.__init__",
        replacement=f"{BASE}.nonblocking_h2d.nonblocking_h2d_wrapper",
    ),
)

H2D_PREFETCH_STREAM = group(
    "lerobot_act.h2d_prefetch_stream",
    replace(
        target="accelerate.data_loader.DataLoaderShard.__iter__",
        replacement=f"{BASE}.h2d_prefetch_stream.prefetch_dataloader_iter",
        runtime_condition=f"{BASE}.h2d_prefetch_stream.supports_prefetch_dataloader",
    ),
    depends_on=("lerobot_act.nonblocking_h2d",),
)

DEFERRED_METRICS = group(
    "lerobot_act.deferred_metrics",
    wrap(
        target=f"{ACT}.ACTPolicy.forward",
        replacement=f"{BASE}.deferred_metrics.deferred_metrics_policy_wrapper",
    ),
    wrap(
        target="lerobot.common.wandb_utils.WandBLogger.log_dict",
        replacement=f"{BASE}.deferred_metrics.deferred_metrics_wandb_wrapper",
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
        runtime_condition=f"{BASE}.infinite_batch_sampler.supports_resume_epoch",
    ),
    wrap(
        target="accelerate.data_loader.BatchSamplerShard.__iter__",
        replacement=f"{BASE}.infinite_batch_sampler.continuous_batches_wrapper",
        runtime_condition=f"{BASE}.infinite_batch_sampler.supports_continuous_batches",
    ),
)

APEX_FUSED_ADAMW = group(
    "lerobot_act.apex_fused_adamw",
    wrap(
        target="lerobot.optim.optimizers.AdamWConfig.build",
        replacement=f"{BASE}.apex_fused_adamw.apex_fused_adamw_wrapper",
    ),
)
