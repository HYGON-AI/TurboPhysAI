# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""TurboPhysAI catalog for OpenPI optimization groups."""

from turbo_physai import group, replace, wrap

CATALOG_GROUPS = (
    "openpi.numa",
    "openpi.network",
    "openpi.ddp_static_fast_path",
    "openpi.compile_stack",
    "openpi.gemma_checkpoint_stride",
    "openpi.fused_adamw",
    "openpi.deferred_metric_sync",
    "openpi.disable_tqdm",
    "openpi.inductor_settings",
    "openpi.gemma_gate_lookup",
    "openpi.siglip_gelu_lookup",
    "openpi.rope_trig_cache",
    "openpi.softmax_forward_schedule",
    "openpi.softmax_backward_reuse",
    "openpi.clip_grad_scale",
    "openpi.batched_adamw",
)

NUMA = group(
    "openpi.numa",
    wrap(
        target="scripts.train_pytorch.setup_ddp",
        replacement="turbo_physai.optimizations.models.openpi.numa.wrap_process_group_numa",
    ),
)

NETWORK = group(
    "openpi.network",
    wrap(
        target="torch.distributed.init_process_group",
        replacement="turbo_physai.optimizations.models.openpi.network.wrap_process_group_network",
    ),
)

DDP_STATIC_FAST_PATH = group(
    "openpi.ddp_static_fast_path",
    wrap(
        target="torch.nn.parallel.DistributedDataParallel.__init__",
        replacement="turbo_physai.optimizations.models.openpi.ddp_static_fast_path.wrap_ddp_static_fast_path",
    ),
)

COMPILE_STACK = group(
    "openpi.compile_stack",
    wrap(
        target="scripts.train_pytorch.init_logging",
        replacement="turbo_physai.optimizations.models.openpi.ddp_compile_mode_main.wrap_init_logging_ddp_compile_mode",
    ),
    wrap(
        target="openpi.models_pytorch.pi0_pytorch.PI0Pytorch._apply_checkpoint",
        replacement="turbo_physai.optimizations.models.openpi.pi0_checkpoint_main.wrap_pi0_main_apply_checkpoint",
    ),
    wrap(
        target="turbo_physai.optimizations.models.openpi.targets.PI0PytorchImageAlias._apply_checkpoint",
        replacement="turbo_physai.optimizations.models.openpi.pi0_checkpoint_image.wrap_pi0_image_apply_checkpoint",
    ),
)

GEMMA_CHECKPOINT_STRIDE = group(
    "openpi.gemma_checkpoint_stride",
    wrap(
        target="openpi.models_pytorch.gemma_pytorch.PaliGemmaWithExpertModel.forward",
        replacement="turbo_physai.optimizations.models.openpi.gemma_checkpoint_stride.wrap_gemma_forward",
    ),
)

FUSED_ADAMW = group(
    "openpi.fused_adamw",
    wrap(
        target="torch.optim.AdamW",
        replacement="turbo_physai.optimizations.models.openpi.fused_adamw.wrap_fused_adamw",
    ),
)

DEFERRED_METRIC_SYNC = group(
    "openpi.deferred_metric_sync",
    replace(
        target="scripts.train_pytorch.train_loop",
        replacement="turbo_physai.optimizations.models.openpi.deferred_metric_sync.deferred_metric_train_loop",
    ),
)

DISABLE_TQDM = group(
    "openpi.disable_tqdm",
    wrap(
        target="scripts.train_pytorch.tqdm.tqdm",
        replacement="turbo_physai.optimizations.models.openpi.disable_tqdm.wrap_tqdm_disabled",
    ),
)

INDUCTOR_SETTINGS = group(
    "openpi.inductor_settings",
    wrap(
        target="torch.compile",
        replacement="turbo_physai.optimizations.models.openpi.inductor_settings.wrap_torch_compile",
    ),
)

# The following groups port the audited OpenPI runtime patch into Turbo's
# ordinary replacement path. Runtime installers are separate groups so each
# optimization can be ablated independently without a patch runner.
GEMMA_GATE_LOOKUP = group(
    "openpi.gemma_gate_lookup",
    wrap(
        target="scripts.train_pytorch.set_seed",
        replacement="turbo_physai.optimizations.models.openpi.gemma_gate_lookup.wrap_gemma_gate_lookup",
    ),
)

SIGLIP_GELU_LOOKUP = group(
    "openpi.siglip_gelu_lookup",
    wrap(
        target="scripts.train_pytorch.build_datasets",
        replacement="turbo_physai.optimizations.models.openpi.siglip_gelu_lookup.wrap_siglip_gelu_lookup",
    ),
)

ROPE_TRIG_CACHE = group(
    "openpi.rope_trig_cache",
    wrap(
        target="openpi.models_pytorch.pi0_pytorch.PI0Pytorch.__init__",
        replacement="turbo_physai.optimizations.models.openpi.rope_trig_cache.wrap_rope_trig_cache",
    ),
)

SOFTMAX_FORWARD_SCHEDULE = group(
    "openpi.softmax_forward_schedule",
    wrap(
        target="openpi.models_pytorch.gemma_pytorch.PaliGemmaWithExpertModel.__init__",
        replacement="turbo_physai.optimizations.models.openpi.softmax_forward_schedule.wrap_softmax_forward_schedule",
    ),
)

SOFTMAX_BACKWARD_REUSE = group(
    "openpi.softmax_backward_reuse",
    wrap(
        target="openpi.models_pytorch.pi0_pytorch.PI0Pytorch.gradient_checkpointing_enable",
        replacement="turbo_physai.optimizations.models.openpi.softmax_backward_reuse.wrap_softmax_backward_reuse",
    ),
)

CLIP_GRAD_SCALE = group(
    "openpi.clip_grad_scale",
    wrap(
        target="torch.nn.utils.clip_grad_norm_",
        replacement="turbo_physai.optimizations.models.openpi.clip_grad_scale.wrap_clip_grad_scale",
    ),
)

BATCHED_ADAMW = group(
    "openpi.batched_adamw",
    wrap(
        target="torch.optim.adamw.AdamW.step",
        replacement="turbo_physai.optimizations.models.openpi.batched_adamw.wrap_batched_adamw",
    ),
)
