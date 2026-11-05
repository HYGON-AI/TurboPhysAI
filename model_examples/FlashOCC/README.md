# FlashOCC 应用说明

本文说明如何在 HCU 环境中，基于官方 FlashOCC 仓库使用 TurboPhysAI 优化。TurboPhysAI 的镜像获取、容器创建和源码安装方法见[安装指南](../../docs/zh/get_started/installation.md)。

## 1. 模型简介

[FlashOCC](https://github.com/Yzichen/FlashOCC) 是面向自动驾驶场景的相机视觉占用预测模型，可从多摄像头图像预测场景中的三维语义占用。

## 2. 优化接入基线

本接入基于官方 FlashOCC 仓库 commit `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd`，TurboPhysAI 优化实现位于 [FlashOCC 优化目录](../../turbo_physai/optimizations/models/flashocc)。

启动时通过 `--optimization-config` 和 `--runtime-config` 显式指定该模型的配置；不需要修改官方 FlashOCC 源码。配置文件的作用说明见 [OptimizationConfig 文档](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig 文档](../../docs/zh/user_guide/runtime_config.md)。

## 3. 准备运行环境与源码

### 3.1 创建容器

宿主机准备好 `autonomous-driving:torch2.10-dtk26.04` 镜像后，执行：

```bash
docker run -d -t \
  -v /opt/hyhal:/opt/hyhal:ro \
  --workdir /workspace/FlashOCC_TurboPhysAI \
  --privileged \
  --shm-size=128G \
  --device=/dev/kfd \
  --device=/dev/dri/ \
  --network=host \
  --device=/dev/mkfd \
  --group-add video \
  --name flashocc_cuiqi_zijia_base_v3 \
  --entrypoint /bin/bash \
  autonomous-driving:torch2.10-dtk26.04 \
  -lc 'exec sleep infinity'

docker exec -it flashocc_cuiqi_zijia_base_v3 bash
```

### 3.2 激活环境并设置工作目录

进入容器后执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "/workspace/FlashOCC_TurboPhysAI"
cd "/workspace/FlashOCC_TurboPhysAI"
```

### 3.3 拉取官方 FlashOCC 源码

```bash
git clone https://github.com/Yzichen/FlashOCC.git "/workspace/FlashOCC_TurboPhysAI/FlashOCC"
git -C "/workspace/FlashOCC_TurboPhysAI/FlashOCC" checkout --detach \
  449dc7947d6549ccb72a99d6f6d8c02f166ec0dd
```

### 3.4 拉取并安装 TurboPhysAI

以下源码安装步骤适用于已准备好 HCU 开发环境的情况，依赖要求和安装检查见[安装指南](../../docs/zh/get_started/installation.md#从源码安装)。

```bash
cd "/workspace/FlashOCC_TurboPhysAI"
git clone --branch main \
  https://github.com/HYGON-AI/TurboPhysAI.git \
  "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI"

cd "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI"
python -c 'import torch; print(torch.__version__, torch.version.hip)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help

```

### 3.5 准备 MMDetection3D 依赖并构建 HCU 扩展

```bash
mkdir -p "/workspace/FlashOCC_TurboPhysAI/runtime"
git clone --branch v1.0.0rc4 \
  https://github.com/open-mmlab/mmdetection3d.git \
  "/workspace/FlashOCC_TurboPhysAI/FlashOCC/mmdetection3d"
git -C "/workspace/FlashOCC_TurboPhysAI/FlashOCC/mmdetection3d" checkout --detach \
  c9541b0db89498fdea5cafd05b7b17f7b625b858

export PYTHONPATH="/workspace/FlashOCC_TurboPhysAI/TurboPhysAI:/workspace/FlashOCC_TurboPhysAI/FlashOCC/mmdetection3d:/workspace/FlashOCC_TurboPhysAI/FlashOCC${PYTHONPATH:+:$PYTHONPATH}"
bash "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/scripts/build_flashocc_ops_external.sh"
```

## 4. 准备 nuScenes 数据并生成标注

按照 [FlashOCC 官方安装文档 Step 3](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) 下载并准备 nuScenes。将数据放到 `/workspace/FlashOCC_TurboPhysAI/FlashOCC/data/nuscenes`，至少包括以下目录：

```text
/workspace/FlashOCC_TurboPhysAI/FlashOCC/data/nuscenes/
├── v1.0-trainval/
├── sweeps/
├── samples/
├── gts/
└── maps/
```

在 FlashOCC 根目录运行官方数据转换脚本，生成训练所需的 PKL 标注：

```bash
cd "/workspace/FlashOCC_TurboPhysAI/FlashOCC"
python tools/create_data_bevdet.py
```

训练脚本读取以下标注文件：

```text
/workspace/FlashOCC_TurboPhysAI/FlashOCC/data/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl
└── bevdetv2-nuscenes_infos_val.pkl
```

## 5. 准备预训练权重

从 [BEVDet 官方模型列表](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/README.md#nuscenes-detection) 的 `BEVDet-R50-CBGS` 下载权重，并将其放到训练脚本使用的 `/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth`：

```bash
mkdir -p "/workspace/FlashOCC_TurboPhysAI/static_data/ckpts"
```

准备完成后的主要目录如下：

```text
/workspace/FlashOCC_TurboPhysAI/
├── FlashOCC/
│   ├── data/nuscenes/
│   └── mmdetection3d/
├── runtime/
├── TurboPhysAI/
└── static_data/
```

## 6. 启动训练

FlashOCC 的优化配置和运行环境配置分别位于：

- `/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml`
- `/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml`

以下命令通过 `--set` 指定当前容器的模型路径、设备和单机网卡，使用单机八卡、单卡 batch size 4（全局 batch size 32）和每卡 24 个数据加载 worker。其他环境需调整这些路径、设备、网卡和进程数，多机运行时不能使用 `lo` 网卡。

通过 `turbo-physai run` 显式加载配置并启动训练：

```bash
cd "/workspace/FlashOCC_TurboPhysAI/FlashOCC"

turbo-physai run \
  --optimization-config "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml" \
  --runtime-config "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml" \
  --set HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --set NCCL_SOCKET_IFNAME=lo \
  --set GLOO_SOCKET_IFNAME=lo \
  --set FLASHOCC_ROOT=/workspace/FlashOCC_TurboPhysAI/FlashOCC \
  --set FLASHOCC_RUNTIME_ROOT=/workspace/FlashOCC_TurboPhysAI/runtime \
  --log-report \
  -- \
  torchrun --nproc_per_node=8 tools/train.py \
    ./projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "/workspace/FlashOCC_TurboPhysAI/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)" \
    --cfg-options \
      data.samples_per_gpu=4 \
      data.workers_per_gpu=24 \
      "load_from=/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth"
```

NUMA 配置、优化应用结果及其检查方法见 [快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
