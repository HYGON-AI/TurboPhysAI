# OpenWAM 运行指南

## 环境准备

### 拉取镜像及创建容器
```
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10

docker run -it --network=host --name=openwam_hcu --privileged --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=512G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1  -v /opt/hyhal:/opt/hyhal:ro  harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10
```

### 克隆代码

```bash
git clone --branch main https://github.com/HYGON-AI/TurboPhysAI.git
git clone https://github.com/OpenWAM-Official/OpenWAM.git
cd OpenWAM
git switch -c turbo-baseline 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995
```

### 安装

在已准备好训练依赖的 Python 环境中安装：

```bash
pip install -e . && pip install numpy==1.25.0
cd ../TurboPhysAI
pip install --upgrade  wheel pip && pip install "setuptools>=64.0,<80.0"
MAX_JOBS=8 pip install -e . --no-build-isolation --no-deps
cd ../OpenWAM
```

后续命令均在 `OpenWAM` 根目录执行，`TurboPhysAI` 为相邻目录。

## 数据与模型权重

按官方 [Assets Preparation](https://github.com/OpenWAM-Official/OpenWAM/blob/0c9ed80d80af1d0c5f4148afe65cc7f462ec3995/README.md#assets-preparation) 准备 RoboTwin 2.0 数据集和 OpenWAM-Alpha foundation checkpoint。checkpoint 目录需包含 `config.yaml`、`checkpoint_step_*.safetensors` 和配套 tokenizer。

以下沿用原启动脚本的路径，均相对于 `OpenWAM` 根目录；资产位于其他目录时，修改启动命令中的对应参数：

| 内容 | 路径 |
| --- | --- |
| RoboTwin 2.0 clean50 数据根目录 | `assets/benchmark_data/robotwin2.0/dataset` |
| Foundation checkpoint | `assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model` |

数据目录应保留官方结构，例如：

```text
dataset/
└── blocks_ranking_size/
    └── aloha-agilex_clean_50/
        └── data/
            ├── episode0.hdf5
            └── ...
```

`dataloader.dataset_dir` 指向上面的 `dataset` 目录，`dataloader.variant=clean_50` 选择 clean50 数据。归一化统计由 reader 自动查找或生成，无需提前创建 `outputs/hcu_alpha_20260930` 目录。仅需复用已有统计文件时，才追加 `+dataloader.normalization_stats_path=/path/to/stats.npy`。

固定基线默认使用 `dual_system/joint_self_attn`、`separate_action_dit` 和 `wan22_ti2v_5b`，启动命令无需重复指定这些模型配置。

## 在线训练

在 `OpenWAM` 根目录执行

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam --log-report \
  --optimization-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/optimization.online.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml \
  torchrun --nproc_per_node=8 scripts/train.py \
    training.speed_preset=null \
    dataloader=robotwin \
    dataloader.dataset_dir=assets/benchmark_data/robotwin2.0/dataset \
    dataloader.variant=clean_50 '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.num_epochs=null training.max_steps=300 training.batch_size=16 \
    training.gradient_accumulation_steps=1 training.save_steps=0 \
    training.output_path=outputs/hcu_openwam_turbo \
    project.wandb.project=null project.seed=42
```

按实际卡数调整 `HIP_VISIBLE_DEVICES` 和 `--nproc_per_node`。`save_steps=0` 不保存周期或最终权重。输出目录另设为 `outputs/hcu_openwam_turbo`，便于与原运行记录区分。注意追加 `training.speed_preset=null` 避免使用官方针对 CUDA 的 cuDNN SDPA 预设。

### 抓取 Trace (Profiling)
如需在训练过程中抓取 PyTorch 算子与通信 Trace 进行性能对比分析，添加环境变量即可自动触发：
- `enable_profiling=1`（或 `true`）：仅对 rank 0 进行采样（默认采样第 15 步），自动输出压缩 Trace 到 `<output_path>/profiler_traces/rank0_trace.json.gz`。
- `enable_profiling=all`：对所有 rank 进行采样。
- `OPENWAM_PROFILE_LIGHT=1`：可选，开启轻量采样模式（关闭 call stack 和 tensor shapes 记录以减少 profiling 开销）。
内置 DCU 专用的 trace sanitize 清洗逻辑，确保 GPU Stream 0 与 CPU 背景线程严格隔离。

## 优化配置

`turbo-physai run --model openwam` 默认加载基础优化；上面的命令显式选择随包在线优化配置，启用编译、SAC 和通信重叠等选项。

| 配方 | 用途 |
| --- | --- |
| [recipe.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.yaml) | 默认基础优化 |
| [recipe.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml) | 在线编码、编译、SAC、通信重叠等优化 |

官方 PR #58 原生支持了分层激活重算 `training.gradient_checkpointing_layers`（`null` 重算全部层；非负整数 N 表示前 N 层重算，例如 `training.gradient_checkpointing_layers=24`；`0` 表示全不重算），原 `checkpoint_skip_layers` 已被官方配置取代。

需要保存自定义配方时，复制 recipe 后编辑，再生成配置。recipe 不能直接传给 `--optimization-config`。

```bash
mkdir -p ../openwam-configs
cp ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml \
  ../openwam-configs/recipe.custom.yaml
# 编辑 ../openwam-configs/recipe.custom.yaml 后执行：
turbo-physai optimization generate \
  --recipe ../openwam-configs/recipe.custom.yaml \
  --repo . --commit 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995 \
  --output ../openwam-configs/optimization.custom.yaml
```

启动时将 `--optimization-config` 改为 `../openwam-configs/optimization.custom.yaml`，其余参数沿用在线训练命令。

仅在修改 recipe 时需要生成配置，生成时 OpenWAM 工作区需保持干净。生成的 YAML 与同目录下的 `.generation.json` 凭据需一起保留；重复生成时添加 `--force`。

在线和缓存优化组合要求原生 Wan22Ti2v VAE38/T5、ZeRO-2、无 CPU optimizer offload、梯度累积为 1、DataLoader workers 大于 0，开启 gradient checkpointing 且关闭 checkpoint offload。Turbo 路径尚待 HCU 多卡正确性和性能验证。

BF16 训练的 ZeRO-2 owner 归约默认使用 BF16 通信，保留已有的显式 `communication_data_type` 设置。

在线和缓存命令均使用 [runtime.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml)，包含针对 DCU 优化的环境变量（如 `OPENWAM_SDPA_CUDNN=0`，`OPENWAM_FAST_INPUT=1`，`OPENWAM_FAST_CAT=1`，`OPENWAM_COMPILE_BLOCKS=1` 等）；需要在目标环境启用 16 个硬件队列时，可在启动前设置 `GPU_MAX_HW_QUEUES=16`。hipBLASLt 库包及配套数据需由目标环境提供；ZeRO-2 overlap 的 HIP BLASLt workspace 默认为 32 MiB，保留用户显式设置的 `CUBLASLT_WORKSPACE_SIZE` 或 `HIPBLASLT_WORKSPACE_SIZE`。

## 优化分组

模型优化通过 Turbo Group 和 `options` 控制，启动环境由 RuntimeConfig 配置，不读取 `OPENWAM_OPT_*`。

| Group | 内容 | 主要 options |
| --- | --- | --- |
| `openwam.data` | RoboTwin 任务筛选和离线 latent 读取 | `vae_latent_cache`, `vae_latent_cache_dir` |
| `openwam.compute` | DCU Norm、投影、SAC 算子边界 | `lightop_norm`, `linear_bias2d` |
| `openwam.architecture` | 文本/本体感知 mask 接入、训练常量、latent 输入和连续 checkpoint 张量 | `constant_cache` |
| `openwam.mot` | 选择性重算（SAC） | `sac_attention`, `sac_ffn_full`, `sac_gemm`, `sac_layers`, `sac_extra_ops` |
| `openwam.text` | 离线 T5 缓存、FA2 padding plan 与 varlen cross-attention | `text_cache_dir`, `padding_plan_cpu`, `fa2_padding`, `fa2_mask_fused` |
| `openwam.training` | ZeRO 参数、ReduceScatter、在线编码/deferred gather、fused AdamW、Host 异步性能路径与 DCU NUMA 绑定 | `zero_overlap`, `zero_reduce_scatter`, `zero_scaled_pack`, `zero_norm_overlap`, `online_encode`, `deferred_gather`, `fused_adamw`, `host_async`, `hw_queues` |

官方 PR #58 原生支持了 VAE 全视频段编码及 channels-last、MoT Triton fast cat/split、Triton fused RoPE、MoT 块编译以及在线 text cache 等通用优化，相应冗余代码已精简，重点保留 Hygon DCU 专属加速特性。

## 离线缓存

沿用在线训练的两份配置。`optimization.online.yaml` 默认关闭 VAE/T5 缓存，使用时通过 `--option` 指定开关和目录，无需重新生成配置。以下将缓存写入 `assets/openwam_latents_clean_50`，T5 缓存位于其中的 `text` 子目录。

### 生成缓存

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam \
  --optimization-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/optimization.online.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml \
  --option openwam.data.vae_latent_cache_dir=assets/openwam_latents_clean_50 \
  torchrun --nproc_per_node=8 --module \
    turbo_physai.optimizations.models.openwam.cache_vae_latents \
    training.speed_preset=null \
    dataloader=robotwin \
    dataloader.dataset_dir=assets/benchmark_data/robotwin2.0/dataset \
    dataloader.variant=clean_50 '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.batch_size=16
```

缓存入口会遍历所选任务的全部训练窗口。完成后执行缓存训练，数据、权重和预处理设置必须与生成阶段一致。

### 使用缓存训练

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam --log-report \
  --optimization-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/optimization.online.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml \
  --option openwam.data.vae_latent_cache=true \
  --option openwam.data.vae_latent_cache_dir=assets/openwam_latents_clean_50 \
  torchrun --nproc_per_node=8 scripts/train.py \
    training.speed_preset=null \
    dataloader=robotwin \
    dataloader.dataset_dir=assets/benchmark_data/robotwin2.0/dataset \
    dataloader.variant=clean_50 '+dataloader.tasks=[blocks_ranking_size]' \
    training.finetune_ckpt_path=assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model \
    training.num_epochs=null training.max_steps=300 training.batch_size=16 \
    training.gradient_accumulation_steps=1 training.save_steps=0 \
    training.output_path=outputs/hcu_openwam_cached \
    project.wandb.project=null project.seed=42
```

已有缓存可直接使用，缓存签名会核对数据窗口、相机、尺寸、权重和文本编码设置。

## 来源与许可证

[provenance.json](../../turbo_physai/optimizations/models/openwam/configs/provenance.json) 记录来源提交与函数映射。OpenWAM 及其原生 Wan 派生函数保持 Apache-2.0；新增 Turbo 接入代码为 BSD-3-Clause。详见 [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)。
