# OpenWAM 运行指南

## 环境准备

在已配置 DTK、HCU PyTorch、FlashAttention、DeepSpeed 和 Accelerate 的 Linux HCU 环境中执行。Turbo 构建依赖见[安装指南](../../docs/zh/get_started/installation.md#从源码安装)，在线优化配方还需要匹配当前环境的 Lightop。

### 克隆代码

```bash
git clone --branch main https://github.com/HYGON-AI/TurboPhysAI.git
git clone https://github.com/OpenWAM-Official/OpenWAM.git
cd OpenWAM
git switch -c turbo-baseline 1204dde1b9e9c232131fc99a3f2c6b78330e7f64
```

### 安装

在已准备好训练依赖的 Python 环境中安装，`--no-deps` 保留现有 HCU 平台依赖：

```bash
python -m pip install -e . --no-build-isolation --no-deps
cd ../TurboPhysAI
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python -m pip install -e . --no-build-isolation --no-deps
cd ../OpenWAM
```

后续命令均在 `OpenWAM` 根目录执行，`TurboPhysAI` 为相邻目录。

## 数据与模型权重

按官方 [Assets Preparation](https://github.com/OpenWAM-Official/OpenWAM/blob/1204dde1b9e9c232131fc99a3f2c6b78330e7f64/README.md#assets-preparation) 准备 RoboTwin 2.0 数据集和 OpenWAM-Alpha foundation checkpoint。checkpoint 目录需包含 `config.yaml`、`checkpoint_step_*.safetensors` 和配套 tokenizer。

将下文的 `/path/to/robotwin2.0/dataset` 和 `/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model` 替换为实际路径。示例使用 `dual_system/joint_self_attn`、Wan22-TI2V、ActionDiT 和 RoboTwin 的 `blocks_ranking_size` 任务。

## 在线训练

### 生成配置

```bash
mkdir -p ../openwam-configs
turbo-physai optimization generate \
  --recipe ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml \
  --repo . --commit 1204dde1b9e9c232131fc99a3f2c6b78330e7f64 \
  --output ../openwam-configs/optimization.online.yaml
```

生成时 OpenWAM 工作区需保持干净。生成的 YAML 与同目录下的 `.generation.json` 凭据需一起保留；重复生成时添加 `--force`。

### 启动

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam --log-report \
  --optimization-config ../openwam-configs/optimization.online.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.optimized.yaml \
  torchrun --nproc_per_node=8 scripts/train.py \
    model=dual_system model/video_backbone=wan22_ti2v_5b \
    model/action_backbone=separate_action_dit \
    model.architecture.variant=joint_self_attn \
    dataloader=robotwin \
    dataloader.dataset_dir=/path/to/robotwin2.0/dataset \
    dataloader.variant=clean_50 '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.num_epochs=null training.max_steps=30 training.batch_size=16 \
    training.zero_stage=2 training.offload_optimizer_device=none \
    training.gradient_accumulation_steps=1 training.dataset_num_workers=8 \
    training.use_gradient_checkpointing=true \
    training.use_gradient_checkpointing_offload=false \
    training.save_steps=100 training.save_full_states_for_resume=false \
    training.output_path=../openwam-outputs/online project.wandb.project=null
```

按实际卡数调整 `HIP_VISIBLE_DEVICES` 和 `--nproc_per_node`，按显存调整每卡 `training.batch_size`。30 步用于检查流程，性能统计需排除编译预热。正常结束仍会保存最终 checkpoint。

## 优化配置

| 配方 | 用途 |
| --- | --- |
| [recipe.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.yaml) | 默认基础优化 |
| [recipe.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml) | 在线编码、编译、SAC、通信重叠等优化 |
| [recipe.cache-build.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.cache-build.yaml) | 生成 VAE latent 和 T5 缓存 |
| [recipe.cached.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.cached.yaml) | 使用离线缓存训练 |

如需调整 options，复制相应 recipe 后编辑，再重新生成配置。recipe 不能直接传给 `--optimization-config`。

```bash
cp ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml \
  ../openwam-configs/recipe.custom.yaml
# 编辑 ../openwam-configs/recipe.custom.yaml 后执行：
turbo-physai optimization generate \
  --recipe ../openwam-configs/recipe.custom.yaml \
  --repo . --commit 1204dde1b9e9c232131fc99a3f2c6b78330e7f64 \
  --output ../openwam-configs/optimization.custom.yaml
```

启动时将 `--optimization-config` 改为 `../openwam-configs/optimization.custom.yaml`，其余参数沿用在线训练命令。

在线和缓存优化组合要求原生 Wan22Ti2v VAE38/T5、ZeRO-2、无 CPU optimizer offload、梯度累积为 1、DataLoader workers 大于 0，开启 gradient checkpointing 且关闭 checkpoint offload。Turbo 路径尚待 HCU 多卡正确性和性能验证。

RuntimeConfig 在启动前设置编译调优、硬件队列和 hipBLASLt 偏好；hipBLASLt 库包及配套数据需由目标环境提供。ZeRO-2 overlap 使用 HIP BLASLt 时，workspace 默认限制为 32 MiB，保留用户显式设置的 `CUBLASLT_WORKSPACE_SIZE` 或 `HIPBLASLT_WORKSPACE_SIZE`。

## 优化分组

模型优化通过 Turbo Group 和 `options` 控制，启动环境由 RuntimeConfig 配置，不读取 `OPENWAM_OPT_*`。

| Group | 内容 | 主要 options |
| --- | --- | --- |
| `openwam.data` | uint8 视频上传、RoboTwin 任务筛选和离线 latent 读取 | `uint8_preprocess`, `vae_latent_cache`, `vae_latent_cache_dir` |
| `openwam.compute` | Wan/ActionDiT pointwise、实数 RoPE、Norm、投影、SAC 算子边界 | `rope_real`, `pointwise_compile`, `lightop_norm`, `linear_bias2d` |
| `openwam.architecture` | 文本/本体感知 mask 接入、训练常量、latent 输入和连续 checkpoint 张量 | `constant_cache` |
| `openwam.mot` | 分段 Attention、block compile、尾层取消重算和选择性重算 | `mot_split_attn`, `compile_mot`, `checkpoint_skip_layers`, `sac_attention`, `sac_ffn_full`, `sac_gemm`, `sac_layers`, `sac_extra_ops` |
| `openwam.text` | 冻结 T5 的前缀裁剪、LRU/离线缓存、文本长度与 FA2 padding plan | `text_cache`, `text_cache_size`, `text_cache_dir`, `t5_prefix`, `text_trim`, `padding_plan_cpu`, `fa2_padding`, `fa2_mask_fused` |
| `openwam.vae` | channels-last、因果缓存 layout、VAE pointwise 和 encoder compile | `vae_channels_last`, `vae_pointwise_compile`, `compile_vae` |
| `openwam.training` | ZeRO 参数、ReduceScatter、在线编码/deferred gather、fused AdamW、Host 异步性能路径 | `zero_overlap`, `zero_reduce_scatter`, `zero_scaled_pack`, `zero_norm_overlap`, `online_encode`, `deferred_gather`, `fused_adamw`, `host_async`, `hw_queues` |

默认配置启用数据/文本/layout/实数 RoPE 和 ZeRO overlap 设置；compile、SAC、通信调度改动、Lightop、fused AdamW、磁盘缓存默认关闭。这是待目标环境验证的接入配置，不代表新的最优性能组合。

`openwam.mot` 依赖 `openwam.compute`；`openwam.training` 依赖其余六组，以保证所需调用入口一起安装。Group 内各 options 可独立配置。

## 离线缓存

### 生成两份配置

```bash
cp ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/recipe.cache-build.yaml \
  ../openwam-configs/recipe.cache-build.yaml
cp ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/recipe.cached.yaml \
  ../openwam-configs/recipe.cached.yaml
```

将两份 recipe 中 `openwam.data.options.vae_latent_cache_dir` 均改为同一个实际绝对路径，例如 `/data/openwam-cache`。生成配方保持 `vae_latent_cache: false`，训练配方保持 `true`；两份配方的 `text_cache` 均为 `true`，T5 缓存默认位于该目录的 `text` 子目录。

```bash
turbo-physai optimization generate \
  --recipe ../openwam-configs/recipe.cache-build.yaml \
  --repo . --commit 1204dde1b9e9c232131fc99a3f2c6b78330e7f64 \
  --output ../openwam-configs/optimization.cache-build.yaml

turbo-physai optimization generate \
  --recipe ../openwam-configs/recipe.cached.yaml \
  --repo . --commit 1204dde1b9e9c232131fc99a3f2c6b78330e7f64 \
  --output ../openwam-configs/optimization.cached.yaml
```

### 生成缓存

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam \
  --optimization-config ../openwam-configs/optimization.cache-build.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.optimized.yaml \
  torchrun --nproc_per_node=8 --module \
    turbo_physai.optimizations.models.openwam.cache_vae_latents \
    model=dual_system model/video_backbone=wan22_ti2v_5b \
    model/action_backbone=separate_action_dit \
    model.architecture.variant=joint_self_attn \
    dataloader=robotwin \
    dataloader.dataset_dir=/path/to/robotwin2.0/dataset \
    dataloader.variant=clean_50 '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model
```

缓存入口会遍历所选任务的全部训练窗口。完成后执行缓存训练，数据、权重和预处理设置必须与生成阶段一致。

### 使用缓存训练

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam --log-report \
  --optimization-config ../openwam-configs/optimization.cached.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.optimized.yaml \
  torchrun --nproc_per_node=8 scripts/train.py \
    model=dual_system model/video_backbone=wan22_ti2v_5b \
    model/action_backbone=separate_action_dit \
    model.architecture.variant=joint_self_attn \
    dataloader=robotwin \
    dataloader.dataset_dir=/path/to/robotwin2.0/dataset \
    dataloader.variant=clean_50 '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.num_epochs=null training.max_steps=30 training.batch_size=16 \
    training.zero_stage=2 training.offload_optimizer_device=none \
    training.gradient_accumulation_steps=1 training.dataset_num_workers=8 \
    training.use_gradient_checkpointing=true \
    training.use_gradient_checkpointing_offload=false \
    training.save_steps=100 training.save_full_states_for_resume=false \
    training.output_path=../openwam-outputs/cached project.wandb.project=null
```

缓存训练默认尾部 24 层不重算，需确认目标显存足够；在线配方为尾部 12 层不重算。`sac_layers: null` 表示对其余 checkpoint 层全部使用所选 SAC 策略。缓存签名会核对数据窗口、相机、尺寸、权重和文本编码设置。

## 来源与许可证

[provenance.json](../../turbo_physai/optimizations/models/openwam/configs/provenance.json) 记录来源提交与函数映射。OpenWAM 及其原生 Wan 派生函数保持 Apache-2.0；新增 Turbo 接入代码为 BSD-3-Clause。详见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
