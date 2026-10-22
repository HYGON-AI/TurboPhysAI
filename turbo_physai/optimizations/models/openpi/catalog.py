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
