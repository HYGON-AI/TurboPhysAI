# OpenWAM 运行指南

## 环境准备

### 拉取镜像及创建容器

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10

docker run -it --network=host --name=openwam_hcu --privileged --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=512G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1  -v /opt/hyhal:/opt/hyhal:ro  harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10
```

### 克隆代码

```bash
git clone --branch feat/openwam https://github.com/iMountTai/TurboPhysAI.git
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

`dataloader.dataset_dir` 指向数据集根目录，`dataloader.variant=clean_50` 选择 clean50 数据。归一化统计由 reader 自动查找或生成，复用已有文件时可追加 `+dataloader.normalization_stats_path=/path/to/stats.npy`。

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

按实际卡数调整 `HIP_VISIBLE_DEVICES` 和 `--nproc_per_node`。保留 `training.speed_preset=null`，让 Turbo 的 RuntimeConfig 决定启动环境。`training.gradient_checkpointing_layers=N` 对前 N 层启用激活重算，默认 `null` 表示全部层，`0` 表示关闭。

## 优化配置

在线训练和缓存训练共用 [optimization.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/optimization.online.yaml) 与 [runtime.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml)。缓存训练通过 `--option` 开启 VAE/T5 缓存并关闭在线编码预取。

优化项由 Turbo Group 的 `options` 控制，定义见 [recipe.online.yaml](../../turbo_physai/optimizations/models/openwam/configs/recipe.online.yaml)。临时调整使用 `--option GROUP.KEY=VALUE`，环境变量使用 `--set KEY=VALUE`。修改 recipe 后需通过 `turbo-physai optimization generate` 重新生成配置，recipe 不能直接传给 `--optimization-config`。生成的 YAML 与同目录隐藏文件 `.optimization*.yaml.generation.json` 一起保留。

训练组合要求原生 Wan22Ti2v VAE38/T5、ZeRO-2、梯度累积为 1、DataLoader workers 大于 0，并开启 gradient checkpointing、关闭 checkpoint offload 和 CPU optimizer offload。HCU 多卡正确性和性能需在目标环境验证。

BF16 owner 归约默认使用 BF16 通信。HIP BLASLt workspace 默认为 32 MiB，保留显式设置的 `communication_data_type`、`CUBLASLT_WORKSPACE_SIZE` 和 `HIPBLASLT_WORKSPACE_SIZE`。

## 离线缓存

缓存生成使用 `optimization.cache-build.yaml`，训练使用 `optimization.online.yaml` 加缓存参数。默认 VAE 缓存在 `assets/openwam_latents_clean_50`，T5 缓存位于其中的 `text` 子目录。更换缓存根目录时，生成命令通过 `--option` 同时设置 `openwam.data.vae_latent_cache_dir` 和 `openwam.text.text_cache_dir`，训练命令只需设置 `openwam.data.vae_latent_cache_dir`，T5 会读取其 `text` 子目录。

### 生成缓存

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
turbo-physai run --model openwam \
  --optimization-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/optimization.cache-build.yaml \
  --runtime-config ../TurboPhysAI/turbo_physai/optimizations/models/openwam/configs/runtime.online.yaml \
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
  --option openwam.training.online_encode=false \
  --set TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 \
  torchrun --nproc_per_node=8 scripts/train.py \
    training.speed_preset=null \
    dataloader=robotwin \
    training.gradient_checkpointing_layers=2 \
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
