# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""TurboPhysAI catalog for OpenPI optimization groups."""

from turbo_physai import group, wrap

CATALOG_GROUPS = (
    "openpi.compile_stack",
    "openpi.gemma_checkpoint_stride",
    "openpi.fused_adamw",
    "openpi.disable_tqdm",
)

COMPILE_STACK = group(
    "openpi.compile_stack",
    wrap(
        target="torch.nn.parallel.DistributedDataParallel.__init__",
        replacement="turbo_physai.optimizations.models.openpi.ddp_state.wrap_ddp_static_fast_path",
    ),
    wrap(
        target="openpi.models_pytorch.pi0_pytorch.PI0Pytorch._apply_checkpoint",
        replacement="turbo_physai.optimizations.models.openpi.pi0_compile.wrap_pi0_compile",
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
DISABLE_TQDM = group(
    "openpi.disable_tqdm",
    wrap(
        target="scripts.train_pytorch.tqdm.tqdm",
        replacement="turbo_physai.optimizations.models.openpi.disable_tqdm.wrap_tqdm_disabled",
    ),
)


