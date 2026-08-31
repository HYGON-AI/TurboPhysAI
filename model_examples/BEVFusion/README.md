# BEVFusion 应用说明

本文说明如何在产品镜像中，基于官方 BEVFusion 仓库应用 TurboPhysAI 的 HCU 优化。镜像获取和容器创建方法参见[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)。

## 1. 模型简介

[BEVFusion](https://github.com/mit-han-lab/bevfusion) 是面向自动驾驶多传感器感知的融合模型，在统一的鸟瞰视角（BEV）空间融合相机与激光雷达特征，支持三维目标检测和 BEV 地图分割等任务。

## 2. 优化接入基线

TurboPhysAI 的 BEVFusion 优化基于官方仓库 commit `326653dc06e0938edf1aae7d01efcd158ba83de5` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/mit-han-lab/bevfusion.git model/BEVFusion
cd model/BEVFusion
git checkout --detach 326653dc06e0938edf1aae7d01efcd158ba83de5
mkdir -p anno_pkls ckpts runs
```

也可以将准备好的官方 BEVFusion 仓库放入宿主机工作目录的 `model/BEVFusion`。下文命令均在容器内的 `/workspace/model/BEVFusion` 执行。产品镜像提供 TurboPhysAI、HCU 运行环境和 BEVFusion 训练环境，不包含模型源码及其源码扩展。

## 4. 安装模型源码扩展

BEVFusion 仓库包含训练所需的 MMDetection3D 源码和原生扩展。首次使用挂载的模型仓库时，在镜像提供的 BEVFusion Conda 环境中完成安装：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate bevfusion
cd /workspace/model/BEVFusion
python -m pip install -e . --no-build-isolation --no-deps
```

该命令将源码包安装到当前 Conda 环境，并编译训练所需的原生扩展。

## 5. 准备 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download) 获取并接受许可后，按照 [BEVFusion 官方数据准备说明](https://github.com/mit-han-lab/bevfusion/blob/main/docs/prepare_dataset.md) 准备 nuScenes 数据、地图扩展和训练所需的 info 文件。本文命令将准备完成的数据目录挂载为 `/workspace/model/BEVFusion/anno_pkls`，启动训练时通过 `--dataset_root ./anno_pkls/` 传入该路径。

数据目录必须同时包含预处理生成的元数据，以及元数据所引用的相机、激光雷达和地图文件。启动训练前可先确认：

```bash
cd /workspace/model/BEVFusion
find anno_pkls -maxdepth 2 -type f | head
```

TurboPhysAI 不下载、转换或重新分发 nuScenes 数据。

## 6. 准备预训练权重

Camera-LiDAR 配置使用 Swin Transformer 相机 Backbone 预训练权重和 LiDAR-only 检测权重。将 BEVFusion 官方文档提供的权重放入以下位置：

```text
ckpts/
├── swint-nuimages-pretrained.pth
└── lidar-only-det.pth
```

也可以执行官方仓库提供的下载脚本，再将对应文件复制或链接到 `ckpts/`：

```bash
cd /workspace/model/BEVFusion
bash tools/download_pretrained.sh
mkdir -p ckpts
ln -sf ../pretrained/swint-nuimages-pretrained.pth \
  ckpts/swint-nuimages-pretrained.pth
ln -sf ../pretrained/lidar-only-det.pth \
  ckpts/lidar-only-det.pth
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 BEVFusion 优化。

### 7.1 单机八卡训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate bevfusion
cd /workspace/model/BEVFusion

turbo-physai run \
  --model bevfusion \
  torchpack dist-run -np 8 \
  python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --run-dir runs/turbo_physai_8card \
    --dataset_root ./anno_pkls/ \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
      ckpts/swint-nuimages-pretrained.pth \
    --load_from ckpts/lidar-only-det.pth \
    --fp16 None
```

`torchpack dist-run -np` 指定当前节点的训练进程数。需要使用自定义交付配置时，通过 `--optimization-config` 和 `--runtime-config` 指定文件。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
