# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""openvla optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.

Declarations only describe target/replacement pairs as strings; nothing here imports
OpenVLA/prismatic and nothing resolves runtime objects at import time.
"""

from __future__ import annotations

from ....engine.definitions import group, replace, wrap

# --- BF16 mixed-precision support detection ---------------------------------
BF16_SUPPORT = group(
    "openvla.bf16_support",
    replace(
        target="prismatic.util.torch_utils.check_bloat16_supported",
        aliases=(
            "prismatic.util.check_bloat16_supported",
            "prismatic.training.strategies.base_strategy.check_bloat16_supported",
        ),
        replacement="turbo_physai.optimizations.models.openvla.fixbf16support.check_bloat16_supported",
    ),
)

# --- FSDP1 per-layer torch.compile -------------------------------------------
COMPILE_FSDP1 = group(
    "openvla.compile.fsdp1",
    wrap(
        target="prismatic.training.strategies.fsdp.FSDPStrategy.run_setup",
        replacement="turbo_physai.optimizations.models.openvla.compile_fsdp1.run_setup_wrapper",
    ),
    wrap(
        target="prismatic.training.strategies.fsdp.FSDPStrategy.save_checkpoint",
        replacement="turbo_physai.optimizations.models.openvla.compile_fsdp1.save_checkpoint_wrapper",
    ),
    wrap(
        target="timm.models.vision_transformer.VisionTransformer._intermediate_layers",
        replacement="turbo_physai.optimizations.models.openvla.vision_timm.timm_intermediate_layers_wrapper",
    ),
)

# --- Fused AdamW (Group enabled => fused is the default) --------------------
FUSED_ADAMW = group(
    "openvla.adamw.fused",
    wrap(
        target="torch.optim.AdamW",
        replacement="turbo_physai.optimizations.models.openvla.fusedAdamW.adamw_fused_wrapper",
    ),
)

# --- FSDP1 communication overlap (limit_all_gathers / fwd+bwd prefetch) ------
FSDP_PREFETCH = group(
    "openvla.fsdp.prefetch",
    wrap(
        target="torch.distributed.fsdp.FullyShardedDataParallel",
        aliases=("prismatic.training.strategies.fsdp.FSDP",),
        replacement="turbo_physai.optimizations.models.openvla.fsdp_prefetch.fsdp_prefetch_wrapper",
    ),
)

# --- Fixed-length (bucketed) text padding ------------------------------------
TEXT_LEN_BUCKET = group(
    "openvla.data.text_len_bucket",
    wrap(
        target="prismatic.util.data_utils.PaddedCollatorForActionPrediction.__call__",
        replacement="turbo_physai.optimizations.models.openvla.text_len_bucket.bucketed_collate_wrapper",
    ),
)

# --- Skip FA2 varlen (unpad) on right-padded prefill -------------------------
SKIP_FA2_UNPAD = group(
    "openvla.llm.skip_fa2_unpad",
    wrap(
        target="transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask",
        replacement="turbo_physai.optimizations.models.openvla.skip_fa2_unpad.make_fast_fa2_causal_mask_wrapper",
    ),
)

# --- RLDS DataLoader: spawned workers (Group options: num_workers, default 1) --
# 仅对 RLDS 数据集（RLDSDataset / EpisodicRLDSDataset）的 DataLoader 强制
# `num_workers=N` + `spawn` context（RLDS 的 TF graph 不能 fork，必须 spawn 重建）；
# RLDS 数据集类包成可 pickle 子类（序列化构造参数、worker 里重建 TF graph）。
DATALOADER_SPAWN = group(
    "openvla.dataloader.spawn",
    wrap(
        target="prismatic.vla.datasets.datasets.RLDSDataset",
        aliases=(
            "prismatic.vla.datasets.RLDSDataset",
            "prismatic.vla.materialize.RLDSDataset",
        ),
        replacement="turbo_physai.optimizations.models.openvla.spawn_dataloader.rlds_dataset_spawn_wrapper",
    ),
    wrap(
        target="prismatic.vla.datasets.datasets.EpisodicRLDSDataset",
        aliases=(
            "prismatic.vla.datasets.EpisodicRLDSDataset",
            "prismatic.vla.materialize.EpisodicRLDSDataset",
        ),
        replacement="turbo_physai.optimizations.models.openvla.spawn_dataloader.episodic_rlds_dataset_spawn_wrapper",
    ),
    wrap(
        target="torch.utils.data.DataLoader",
        aliases=("prismatic.training.strategies.base_strategy.DataLoader",),
        replacement="turbo_physai.optimizations.models.openvla.spawn_dataloader.dataloader_spawn_wrapper",
    ),
)

# --- Reproducibility: a reproducible data stream ------------------------------
# 回答「第 k 步看到哪些样本」。上游只播种 random/numpy/torch；RLDS/dlimp 的四处随机点
# （TFDS 文件级 shuffle、mixure 的 sample_from_datasets、frame shuffle、增强里的
# tf.random.uniform）都以 seed=None 构造，TF 在构建时从全局 RNG 取值，不播种则每次
# 启动采样顺序都不同。本组把 run seed 打进 TF 全局 RNG，并显式送给 mixture 采样与
# frame shuffle —— 也就是已验证分支显式播种的那两处。
#
# frame shuffle 锚在 `dlimp.DLataset.shuffle`，**不是** `tf.data.Dataset.shuffle`：
# TFDS 的文件级 shuffle（`instruction_ds.shuffle(seed=read_config.shuffle_seed)`，
# dlimp 不传 shuffle_seed）走的是后者，一起改掉会改变 shard 读取顺序，曲线就与已验证
# 分支对不上了（eager 下 seed=None 派生自全局种子，同种子即同顺序）。
# `dlimp.DLataset.shuffle` 恰好只命中 `make_interleaved_dataset` 里那次 frame shuffle。
#
# `prismatic.util.set_global_seed` 这个 alias 是必需的：`prismatic/util/__init__.py`
# 在导入时把 set_global_seed 重新绑定进 `prismatic.util`，而 `vla-scripts/train.py`
# 与 `scripts/pretrain.py` 都是从 `prismatic.util` 导入它的。
# `make_interleaved_dataset` 的两个 alias 同样必需：`rlds/__init__.py` 重新导出了它，
# 而 `datasets.py` 在模块导入时把该名字绑进了自己的命名空间（`RLDSDataset.make_dataset`
# 调用的就是这个绑定）。
# options: explicit_seeds（默认 true）、hold_shuffle_permutation（默认 true）。
#
# 与 `openvla.dataloader.spawn` 的配合：spawn worker 里框架 patch 不生效（bootstrap 会
# 跳过 `python -c` 辅助进程），因此每个 worker 的数据流种子由 spawn 重建路径回调
# `reproducibility.worker_seeded_dataset_class` 安装；没有本组发布的环境标记时它是空操作。
DATA_ORDER = group(
    "openvla.reproducibility.data_order",
    wrap(
        target="prismatic.util.torch_utils.set_global_seed",
        aliases=("prismatic.util.set_global_seed",),
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.set_global_seed_wrapper",
    ),
    wrap(
        target="prismatic.util.torch_utils.worker_init_function",
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.worker_init_function_wrapper",
    ),
    wrap(
        target="prismatic.vla.datasets.rlds.dataset.make_dataset_from_rlds",
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.make_dataset_from_rlds_wrapper",
    ),
    wrap(
        target="prismatic.vla.datasets.rlds.dataset.make_interleaved_dataset",
        aliases=(
            "prismatic.vla.datasets.rlds.make_interleaved_dataset",
            "prismatic.vla.datasets.datasets.make_interleaved_dataset",
        ),
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.make_interleaved_dataset_wrapper",
    ),
    wrap(
        target="dlimp.DLataset.sample_from_datasets",
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.sample_from_datasets_wrapper",
    ),
    wrap(
        target="dlimp.DLataset.shuffle",
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.dlimp_shuffle_wrapper",
    ),
)

# --- Reproducibility: consistent kernel selection -----------------------------
# 回答「同样的样本算出同样的数」。种子管不到 kernel 选择：cudnn/MIOpen 的 autotune 可能
# 挑中不同的卷积实现、TF32 截断 fp32 尾数、部分反向归约顺序不定。本组固定这些选择。
#
# 锚点选 `PrismaticVLM.__init__` 的理由：它是训练循环之前最后碰全局 RNG 的地方
# （内部 `torch.manual_seed(vision_backbone.embed_dim)` 会覆盖 run seed，本组在构造后
# 重新落实）, 是每个训练/评测入口都必然经过的点, 且与其它组不冲突（set_global_seed 归
# data_order、run_vla_training 归 gc.freeze、run_setup 归 compile.fsdp1、DataLoader 归
# dataloader.spawn）。容器级确定性（MIOpen / rocBLAS）必须由 RuntimeConfig 在进程启动前
# 设置，不在本组内。
# options: cudnn_deterministic(true)、benchmark(false)、allow_tf32(false)、
#          deterministic_algorithms(true)、strict(false)。
DETERMINISM = group(
    "openvla.reproducibility.determinism",
    wrap(
        target="prismatic.models.vlms.prismatic.PrismaticVLM.__init__",
        replacement="turbo_physai.optimizations.models.openvla.reproducibility.prismatic_vlm_init_wrapper",
    ),
)

# --- Python GC freeze before the training loop -------------------------------
GC_FREEZE = group(
    "openvla.gc.freeze",
    wrap(
        target=(
            "prismatic.training.strategies.base_strategy."
            "TrainingStrategy.run_vla_training"
        ),
        replacement="turbo_physai.optimizations.models.openvla.gc_freeze.gc_freeze_wrapper",
    ),
)
