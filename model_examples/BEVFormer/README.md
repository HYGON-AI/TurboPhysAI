# BEVFormer 应用说明

本文说明如何在产品镜像中，基于官方 BEVFormer 仓库应用 TurboPhysAI 的 HCU 优化。镜像获取和容器创建方法参见[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)。

## 1. 模型简介

[BEVFormer](https://github.com/fundamentalvision/BEVFormer) 是面向自动驾驶纯视觉感知的模型，通过时空 Transformer 从多摄像头图像中学习统一的鸟瞰视角（BEV）表示，用于三维目标检测等感知任务。

## 2. 优化接入基线

TurboPhysAI 的 BEVFormer 优化基于官方仓库 commit `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/fundamentalvision/BEVFormer.git model/BEVFormer
cd model/BEVFormer
git checkout --detach 66b65f3a1f58caf0507cb2a971b9c0e7f842376c
mkdir -p data work_dirs
```

也可以将准备好的官方 BEVFormer 仓库放入宿主机工作目录的 `model/BEVFormer`。下文命令均在容器内的 `/workspace/model/BEVFormer` 执行。产品镜像提供 TurboPhysAI、HCU 运行环境和 BEVFormer 训练依赖，不包含模型源码。

## 4. 准备官方 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download) 获取并接受许可后，下载 nuScenes v1.0 full 数据集和 CAN bus expansion。BEVFormer 的时间序列训练使用 CAN bus 位姿，因此不能只准备相机文件。解压后，宿主机数据目录应为：

```text
/workspace/model/BEVFormer/data/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-trainval/
    └── v1.0-test/
```

容器内生成官方 BEVFormer temporal info 文件：

```bash
cd /workspace/model/BEVFormer

python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data

ls -lh data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl
```

该命令生成 `nuscenes_infos_temporal_train.pkl` 与 `nuscenes_infos_temporal_val.pkl`。训练配置从 `data/nuscenes/` 读取这两个文件。同一命令也会处理 `v1.0-test` 并生成测试 info。

## 5. 准备预训练权重

`BEVFormer-base` 配置默认加载 `ckpts/r101_dcn_fcos3d_pretrain.pth`。将官方 BEVFormer 文档指定的 R101-DCN-FCOS3D 预训练权重下载到该位置：

```bash
cd /workspace/model/BEVFormer
mkdir -p ckpts
wget -O ckpts/r101_dcn_fcos3d_pretrain.pth \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

TurboPhysAI 已安装在镜像的 Python 环境中。后续命令使用 `--model bevformer` 自动选择随组件安装的 OptimizationConfig 和 RuntimeConfig，不依赖固定安装目录。

## 6. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 BEVFormer 优化。

BEVFormer RuntimeConfig 包含训练所需的运行环境和 NUMA 配置。

### 6.1 单机八卡训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate bevformer
cd /workspace/model/BEVFormer

turbo-physai run \
  --model bevformer \
  torchrun --nproc-per-node=8 tools/train.py \
    ./projects/configs/bevformer/bevformer_base.py \
    --launcher pytorch
```

`--nproc-per-node` 指定当前节点的训练进程数。`--model bevformer` 自动加载内置 OptimizationConfig 和 RuntimeConfig。RuntimeConfig 会为每个 rank 自动选择本地 NUMA node。需要使用自定义交付配置时，通过 `--optimization-config` 和 `--runtime-config` 指定对应文件。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
