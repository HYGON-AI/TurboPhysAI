# BEVFormer 应用说明

本文说明如何通过组件附带产品镜像，在官方 BEVFormer 仓库基线上应用 TurboPhysAI 的 HCU
优化。通用操作流程参见[快速开始](../../docs/zh/get_started/quick_start.md)。

## 1. 模型简介

[BEVFormer](https://github.com/fundamentalvision/BEVFormer) 是面向自动驾驶纯视觉感知的模型，
通过时空 Transformer 从多摄像头图像中学习统一的鸟瞰视角（BEV）表示，用于三维目标检测等感知任务。

## 2. 优化接入基线

TurboPhysAI 的 BEVFormer 优化基于官方仓库 commit
`66b65f3a1f58caf0507cb2a971b9c0e7f842376c` 接入。
优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 使用组件附带产品镜像

镜像 `harbor.sourcefind.cn:5443/dcu/admin/base/turbophysai:2.7.1-ubuntu22.04-dtk26.04-py3.10` 已预装 TurboPhysAI、HCU 运行环境及 BEVFormer
训练依赖，并包含经过验证的 OptimizationConfig 和 RuntimeConfig。镜像不包含 BEVFormer
模型源码；用户需要准备官方模型仓库，推荐使用上文优化接入基线，并将模型仓库、数据集和训练输出
目录挂载到容器中。容器内无需重新安装 TurboPhysAI 或模型依赖。

```bash
export IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/turbophysai:2.7.1-ubuntu22.04-dtk26.04-py3.10
export MODEL_ROOT=/path/to/BEVFormer
export DATA_ROOT=/data/bevformer-data
export WORK_ROOT=/data/bevformer-work

docker pull "$IMAGE"
mkdir -p "$DATA_ROOT" "$WORK_ROOT"

git -C "$MODEL_ROOT" rev-parse HEAD
```

`MODEL_ROOT` 必须指向官方 BEVFormer 仓库，且当前 commit 必须为
`66b65f3a1f58caf0507cb2a971b9c0e7f842376c`。在已配置 HCU 容器运行环境的
主机上，挂载模型、数据和工作目录启动镜像：

```bash
docker run -dit \
  --network=host \
  --name=turbophysai-bevformer \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=512G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODEL_ROOT":/workspace/model/BEVFormer \
  -v "$DATA_ROOT":/workspace/model/BEVFormer/data \
  -v "$WORK_ROOT":/workspace/model/BEVFormer/work_dirs \
  "$IMAGE"

docker exec -it turbophysai-bevformer bash
```

下文命令均在容器内的 `/workspace/model/BEVFormer` 执行。模型代码来自宿主机挂载的
官方仓库，TurboPhysAI 和运行依赖由镜像提供。

## 4. 准备官方 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download) 获取并接受许可后，下载
nuScenes v1.0 full 数据集和 CAN bus expansion。BEVFormer 的时间序列训练使用 CAN
bus 位姿，因此不能只准备相机文件。解压后，宿主机数据目录应为：

```text
$DATA_ROOT/
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

该命令生成 `nuscenes_infos_temporal_train.pkl` 与
`nuscenes_infos_temporal_val.pkl`。训练配置从 `data/nuscenes/` 读取这两个文件。
同一命令也会处理 `v1.0-test` 并生成测试 info。

## 5. 准备预训练权重

`BEVFormer-base` 配置默认加载 `ckpts/r101_dcn_fcos3d_pretrain.pth`。将官方 BEVFormer 文档指定
的 R101-DCN-FCOS3D 预训练权重下载到该位置：

```bash
cd /workspace/model/BEVFormer
mkdir -p ckpts
wget -O ckpts/r101_dcn_fcos3d_pretrain.pth \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

TurboPhysAI 已安装在镜像的 Python 环境中。后续命令使用 `--model bevformer`
自动选择随组件安装的 OptimizationConfig 和 RuntimeConfig，不依赖固定安装目录。

## 6. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的
[OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和
[RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用
BEVFormer 优化。

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

`--nproc-per-node` 指定当前节点的训练进程数。
`--model bevformer` 自动加载内置 OptimizationConfig 和 RuntimeConfig。RuntimeConfig
会为每个 rank 自动选择本地 NUMA node。需要使用自定义交付配置时，通过
`--optimization-config` 和 `--runtime-config` 指定对应文件。

优化应用状态和训练结果的确认方法见[快速开始：查看执行结果](../../docs/zh/get_started/quick_start.md#4-查看执行结果)。
