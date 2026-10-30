# FlashOCC 应用说明

本文说明如何在 HCU 环境中，基于官方 FlashOCC 仓库使用 TurboPhysAI 优化。教程包括源码基线、TurboPhysAI 插件安装、外置依赖与扩展构建、数据和权重准备，以及训练启动。

## 1. 模型简介

[FlashOCC](https://github.com/Yzichen/FlashOCC) 是面向自动驾驶场景的相机视觉占用预测模型，可从多摄像头图像预测场景中的三维语义占用。

## 2. 优化接入基线

本接入基于官方 FlashOCC 仓库 commit `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd`，TurboPhysAI 优化实现位于 [`flashocc-optimization`](https://github.com/Coulsonq/TurboPhysAI/tree/flashocc-optimization/turbo_physai/optimizations/models/flashocc)。

启动时通过 `--optimization-config` 和 `--runtime-config` 显式指定该模型的配置；不需要修改官方 FlashOCC 源码。配置文件的作用说明见 [OptimizationConfig 文档](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig 文档](../../docs/zh/user_guide/runtime_config.md)。

## 3. 准备运行环境与源码

以下命令假设在同一个容器、同一个已激活的 shell 中依次执行。镜像需提供 HCU/DTK、PyTorch 和 `mmcv1` Conda 环境。

### 3.1 创建容器

宿主机需预先准备自驾基础镜像 `autonomous-driving:torch2.10-dtk26.04`，然后创建容器：

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
  --name flashocc-turbo \
  --entrypoint /bin/bash \
  autonomous-driving:torch2.10-dtk26.04 \
  -lc 'exec sleep infinity'
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

```bash
cd "/workspace/FlashOCC_TurboPhysAI"
git clone --branch flashocc-optimization \
  https://github.com/HYGON-AI/TurboPhysAI.git \
  "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI"

cd "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI"
python -c 'import torch; print(torch.__version__, torch.version.hip)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help

```

### 3.5 准备外置 MMDetection3D 依赖并构建 HCU 扩展

```bash
mkdir -p "/workspace/FlashOCC_TurboPhysAI/runtime"
git clone --branch v1.0.0rc4 \
  https://github.com/open-mmlab/mmdetection3d.git \
  "/workspace/FlashOCC_TurboPhysAI/runtime/mmdetection3d"
git -C "/workspace/FlashOCC_TurboPhysAI/runtime/mmdetection3d" checkout --detach \
  c9541b0db89498fdea5cafd05b7b17f7b625b858

export PYTHONPATH="/workspace/FlashOCC_TurboPhysAI/TurboPhysAI:/workspace/FlashOCC_TurboPhysAI/runtime/mmdetection3d:/workspace/FlashOCC_TurboPhysAI/FlashOCC${PYTHONPATH:+:$PYTHONPATH}"
bash "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/scripts/build_flashocc_ops_external.sh"
```

准备完成后的主要目录如下：

```text
/workspace/FlashOCC_TurboPhysAI/
├── FlashOCC/
├── runtime/
│   └── mmdetection3d/
├── TurboPhysAI/
└── static_data/
```

## 4. 准备 nuScenes 数据并生成标注

按照 [FlashOCC 官方安装文档 Step 3](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) 下载并准备 nuScenes。将数据放到 `/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes`，至少包括以下目录：

```text
/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/
├── v1.0-trainval/
├── sweeps/
├── samples/
├── gts/
└── maps/
```

在 `static_data` 目录运行官方数据转换脚本，生成训练所需的 PKL 标注：

```bash
mkdir -p "/workspace/FlashOCC_TurboPhysAI/static_data"
cd "/workspace/FlashOCC_TurboPhysAI/static_data"

PYTHONPATH="/workspace/FlashOCC_TurboPhysAI/TurboPhysAI:/workspace/FlashOCC_TurboPhysAI/runtime/mmdetection3d:/workspace/FlashOCC_TurboPhysAI/FlashOCC${PYTHONPATH:+:$PYTHONPATH}" \
python - <<'PY'
import runpy
import sys
import types

tools = types.ModuleType("tools")
tools.__path__ = ["/workspace/FlashOCC_TurboPhysAI/FlashOCC/tools"]
sys.modules["tools"] = tools

runpy.run_path(
    "/workspace/FlashOCC_TurboPhysAI/FlashOCC/tools/create_data_bevdet.py",
    run_name="__main__",
)
PY
```

训练脚本读取以下标注文件：

```text
/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl
└── bevdetv2-nuscenes_infos_val.pkl
```

## 5. 准备预训练权重

下载 `bevdet-r50-cbgs.pth` 到训练脚本使用的 `/workspace/FlashOCC_TurboPhysAI/static_data/ckpts`：

```bash
mkdir -p "/workspace/FlashOCC_TurboPhysAI/static_data/ckpts"
curl -fL --retry 5 --retry-delay 3 \
  -o "/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth" \
  https://haihub-model-bj-1251001002.cos.ap-beijing.myqcloud.com/HyperDrive/ckpt/bevdet-r50-cbgs.pth
```

## 6. 启动训练

FlashOCC 的优化配置和运行环境配置分别位于：

- `/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml`
- `/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml`

当前 `runtime.yaml` 指定使用 HCU 0–7。以下命令使用单机八卡、单卡 batch size 24 和每卡 24 个数据加载 worker。运行前请确认设备空闲且显存充足；使用其他设备时需同步调整 RuntimeConfig 和进程数。

通过 `turbo-physai run` 显式加载配置并启动训练：

```bash
cd "/workspace/FlashOCC_TurboPhysAI/FlashOCC"

turbo-physai run \
  --optimization-config "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml" \
  --runtime-config "/workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml" \
  --log-report \
  -- \
  torchrun --nproc_per_node=8 --master_port=29500 tools/train.py \
    ./projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "/workspace/FlashOCC_TurboPhysAI/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)" \
    --cfg-options \
      data.samples_per_gpu=24 \
      data.workers_per_gpu=24 \
      evaluation.start=24 \
      "data.train.data_root=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/" \
      "data.train.ann_file=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl" \
      "data.val.data_root=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/" \
      "data.val.ann_file=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl" \
      "data.test.data_root=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/" \
      "data.test.ann_file=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl" \
      "load_from=/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth" \
      log_config.interval=1
```

NUMA 配置、优化应用结果及其检查方法见 [快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
