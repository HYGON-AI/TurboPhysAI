# MapTRv2 应用说明

## 1. 模型简介

[MapTRv2](https://github.com/hustvl/MapTR/tree/maptrv2) 是面向自动驾驶在线矢量化地图构建的模型，通过多摄像头图像提取鸟瞰视角（BEV）特征，预测车道分隔线、人行横道和道路边界等地图元素。本文使用 ResNet-50 和 nuScenes 24 epoch 训练配置。

## 2. 优化接入基线

TurboPhysAI 的 MapTRv2 优化基于官方 `maptrv2` 分支 commit `e03f097abef19e1ba3fed5f471a8d80fbfa0a064` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

在 HCU 宿主机上设置镜像、容器名称和资源目录。下面以宿主机的 `/workspace` 和 `/dataset` 为例，分别挂载到容器的相同路径，以便使用下文的模型、权重和 nuScenes 数据路径；如宿主机路径不同，请调整变量。

```bash
export IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/autonomous-driving:ubuntu22.04-dtk26.04-torch2.10-py3.10-20261107
export CONTAINER_NAME=maptrv2
export HOST_WORKSPACE=/workspace
export HOST_DATASET=/dataset

docker pull "$IMAGE"
docker run -dit \
  --name "$CONTAINER_NAME" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --shm-size=128G \
  --workdir=/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_WORKSPACE":/workspace \
  -v "$HOST_DATASET":/dataset \
  "$IMAGE" /bin/bash

docker exec -it "$CONTAINER_NAME" bash
```

MapTRv2 使用 MMCV 1.x、MMDetection 2.x 训练栈。进入上述配套镜像后激活 `mmcv1` 环境，并在该环境中安装 TurboPhysAI；后续模型准备、数据处理和训练均使用同一环境：

```bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1

python -m pip install --upgrade turbo-physai \
  -i https://pypi.sourcefind.cn/nightly/dtk2604/ --no-deps

python - <<'PYTHON'
import torch, mmcv, mmdet, mmdet3d, shapely
import turbo_physai._C, GeometricKernelAttention
from apex.optimizers import FusedAdam
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
print('PyTorch:', torch.__version__, 'MMCV:', mmcv.__version__)
print('MMDetection:', mmdet.__version__, 'MMDetection3D:', mmdet3d.__version__)
print('Shapely:', shapely.__version__)
PYTHON
```

上述导入检查用于确认当前 Python 环境中的组件和原生扩展可加载；`--no-deps` 不会安装这些依赖。若缺少 MMDetection3D 或 `GeometricKernelAttention`，需先参考[模型官方安装说明](https://github.com/hustvl/MapTR/blob/e03f097abef19e1ba3fed5f471a8d80fbfa0a064/docs/install.md)准备匹配当前 HCU/PyTorch 的扩展，克隆模型源码不会自动安装它们。

模型源码、数据和权重需单独准备。以上宿主机目录须实际包含后文所需资源，尤其要保证 `data/nuscenes` 软链接指向的 `/dataset/nuscenes` 在容器中可访问。已有容器需要重新创建才能增加挂载。镜像和安装说明见[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)。

在容器中可通过 `hy-smi` 检查设备，通过 `turbo-physai --help` 检查组件安装。重新进入容器或打开新终端时，需再次激活 `mmcv1`。

## 4. 准备模型源码

```bash
mkdir -p /workspace/model
cd /workspace/model
git clone -b maptrv2 https://github.com/hustvl/MapTR.git MapTrv2
cd /workspace/model/MapTrv2
git checkout --detach e03f097abef19e1ba3fed5f471a8d80fbfa0a064
mkdir -p data ckpts work_dirs
```

也可使用已准备的官方源码，执行 `git rev-parse HEAD` 检查版本。生成 OptimizationConfig 时需要干净的模型工作区。

## 5. 准备 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download)获取数据和许可，按 [MapTR 官方数据准备说明](https://github.com/hustvl/MapTR/blob/maptrv2/docs/prepare_dataset.md)准备数据、地图扩展和 MapTRv2 专用的 train/val info 文件。自行生成 info 文件时还需 CAN bus expansion。已有资源可跳过下载和转换。以下给出数据和验证地图标注的容器内路径示例，运行前请确认文件实际存在：

```text
/workspace/model/MapTrv2/data/nuscenes -> /dataset/nuscenes
├── maps/、samples/、sweeps/、v1.0-trainval/
├── nuscenes_map_infos_temporal_train.pkl
└── nuscenes_map_infos_temporal_val.pkl
/workspace/MapTRv2/anno_pkls/nuscenes_map_anns_val.json
```

若 `data/nuscenes` 尚不存在，在模型目录下创建软链接；已有路径则直接检查其内容：

```bash
cd /workspace/model/MapTrv2
if [ ! -e data/nuscenes ] && [ ! -L data/nuscenes ]; then
  ln -s /dataset/nuscenes data/nuscenes
fi
ls -lh data/nuscenes/nuscenes_map_infos_temporal_{train,val}.pkl
```

验证地图标注不在模型配置默认的数据目录中，下面的训练命令显式指定该路径。其他环境请按实际挂载位置调整。

## 6. 准备预训练权重

R50 配置使用 `ckpts/resnet50-19c8e357.pth`。缺少时可从 PyTorch 下载：

```bash
cd /workspace/model/MapTrv2
mkdir -p ckpts
test -e ckpts/resnet50-19c8e357.pth || wget -O ckpts/resnet50-19c8e357.pth \
  https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## 7. 通过 turbo-physai run 启动训练

`--model maptrv2` 加载已安装 TurboPhysAI 中的 OptimizationConfig 和 RuntimeConfig，请确认安装的组件包含 MapTRv2 优化。修改 recipe 或更换环境后，按[优化配置说明](../../docs/zh/user_guide/optimization_config.md)在干净的模型基线上重新生成配置，并通过 `--optimization-config` 指定。

### 7.1 快速验证（单机八卡）

使用官方训练入口运行 1 个 epoch，并跳过验证集评估。按当前数据和每卡 batch size 12 的配置，约为 294 次迭代；实际次数取决于数据量和设备数。该命令用于检查训练链路，不用于判断完整精度或稳定吞吐。

```bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/model/MapTrv2
export SMOKE_OUTPUT="work_dirs/maptrv2_smoke_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$SMOKE_OUTPUT"

turbo-physai run \
  --model maptrv2 \
  --log-report \
  torchrun --nproc-per-node=8 --master-port=29501 tools/train.py \
    projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
    --launcher pytorch \
    --no-validate \
    --work-dir "$SMOKE_OUTPUT" \
    --cfg-options \
      data.samples_per_gpu=12 \
      total_epochs=1 \
      runner.max_epochs=1 \
      log_config.interval=1 2>&1 | tee "$SMOKE_OUTPUT/train.log"
```

确认命令退出码为 0、日志达到第 1 epoch 的最后一次迭代，并检查优化报告中各组的状态。`applied` 表示优化已安装，实际执行仍取决于运行条件。

### 7.2 单机八卡训练

以下命令使用 8 卡、每卡 batch size 12，并沿用当前模型配置的每卡 4 个 DataLoader worker 和 24 epoch；命令显式指定第 24 epoch 验证、逐 step 输出日志：

```bash
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/model/MapTrv2
export TRAIN_OUTPUT="work_dirs/maptrv2_optimized_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$TRAIN_OUTPUT"

turbo-physai run \
  --model maptrv2 \
  --log-report \
  torchrun --nproc-per-node=8 --master-port=29513 tools/train.py \
    projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
    --launcher pytorch \
    --work-dir "$TRAIN_OUTPUT" \
    --cfg-options \
      data.samples_per_gpu=12 \
      data.val.map_ann_file=/workspace/MapTRv2/anno_pkls/nuscenes_map_anns_val.json \
      evaluation.interval=24 \
      log_config.interval=1 \
    2>&1 | tee "$TRAIN_OUTPUT/train.log"
```

该命令使用已安装组件中的配置，随本示例提供的 `runtime.yaml` 将 `ALLREDUCE_STREAM_WITH_COMPUTE` 设为 `1`；基线日志未记录该变量的实际值。确保端口未被占用，并根据实际 HCU 拓扑调整 RuntimeConfig 的 RCCL、NUMA 参数；需要指定网卡时，可在 `torchrun` 前添加 `--set NCCL_SOCKET_IFNAME=<网卡名>`。

优化应用状态的确认方法见[优化应用报告](../../docs/zh/user_guide/report.md)。
