# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""TurboPhysAI catalog for OpenPI optimization groups."""

from turbo_physai import group, wrap

CATALOG_GROUPS = (
    "openpi.network",
    "openpi.compile_stack",
    "openpi.gemma_checkpoint_stride",
    "openpi.fused_adamw",
    "openpi.disable_tqdm",
    "openpi.inductor_settings",
    "openpi.kernel_runtime",
    "openpi.clip_grad_scale",
)

NETWORK = group(
    "openpi.network",
    wrap(
        target="torch.distributed.init_process_group",
        replacement="turbo_physai.optimizations.models.openpi.network.wrap_process_group_network",
    ),
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

INDUCTOR_SETTINGS = group(
    "openpi.inductor_settings",
    wrap(
        target="torch.compile",
        replacement="turbo_physai.optimizations.models.openpi.inductor_settings.wrap_torch_compile",
    ),
)

KERNEL_RUNTIME = group(
    "openpi.kernel_runtime",
    wrap(
        target="torch._inductor.runtime.triton_heuristics.CachingAutotuner.run",
        replacement="turbo_physai.optimizations.models.openpi.kernel_runtime.wrap_kernel_runtime",
    ),
    wrap(
        target="openpi.models_pytorch.pi0_pytorch.PI0Pytorch.forward",
        replacement="turbo_physai.optimizations.models.openpi.rope_trig_cache.wrap_rope_forward",
    ),
)

CLIP_GRAD_SCALE = group(
    "openpi.clip_grad_scale",
    wrap(
        target="torch.nn.utils.clip_grad._clip_grads_with_norm_",
        replacement="turbo_physai.optimizations.models.openpi.clip_grad_scale.wrap_clip_grad_scale",
    ),
)
