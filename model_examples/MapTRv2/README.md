# MapTRv2 应用说明

## 1. 模型简介

[MapTRv2](https://github.com/hustvl/MapTR/tree/maptrv2) 是面向自动驾驶在线矢量化高精地图构建的模型，通过多摄像头图像提取鸟瞰视角（BEV）特征，利用分层查询与有序点集表示直接预测车道分隔线、人行横道和道路边界等矢量化地图元素，并引入辅助深度监督、BEV/PV 语义分割以及一对多匹配策略以提升训练收敛与检测精度。本文使用 ResNet-50 Backbone 和 nuScenes 24 epoch 训练配置（`projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py`）。

## 2. 优化接入基线

TurboPhysAI 的 MapTRv2 优化基于官方 `maptrv2` 分支 commit `e03f097abef19e1ba3fed5f471a8d80fbfa0a064` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

### 3.1 创建容器

在 HCU 宿主机上使用自动驾驶配套镜像创建容器（完整参数说明见[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)）：

```bash
export IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/autonomous-driving:ubuntu22.04-dtk26.04-torch2.10-py3.10-20261107
export CONTAINER_NAME=maptrv2
export HOST_WORKSPACE=/path/to/workspace
export HOST_DATASET=/path/to/dataset

docker pull "$IMAGE"
docker run -dit \
  --network=host \
  --name "$CONTAINER_NAME" \
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
  --workdir=/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_WORKSPACE":/workspace \
  -v "$HOST_DATASET":/dataset:ro \
  "$IMAGE" /bin/bash

docker exec -it "$CONTAINER_NAME" bash
```

### 3.2 激活 `mmcv1` 环境并安装 TurboPhysAI

MapTRv2 依赖 `mmcv1` 环境（MMCV 1.x / MMDetection 2.x）。进入容器后加载 DTK 环境、激活 `mmcv1` 并安装 TurboPhysAI（下文命令均在容器内同一环境中执行）：

```bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1

python -m pip install --upgrade turbo-physai \
  -i https://pypi.sourcefind.cn/nightly/dtk2604/ --no-deps
```

若从本地挂载的 TurboPhysAI 源码安装，在源码目录执行（先清理旧 `dist/` 避免通配符匹配多个 wheel）：

```bash
git config --global --add safe.directory /workspace/TurboPhysAI
rm -rf dist
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
```

## 4. 准备模型源码与扩展

### 4.1 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone -b maptrv2 https://github.com/hustvl/MapTR.git model/MapTrv2
cd model/MapTrv2
git checkout --detach e03f097abef19e1ba3fed5f471a8d80fbfa0a064
mkdir -p data ckpts work_dirs
git config --global --add safe.directory /workspace/model/MapTrv2
```

### 4.2 安装模型源码扩展

配套镜像的 `mmcv1` 缺少 MapTRv2 仓库自带的 `GeometricKernelAttention` 以及含 `bev_pool` / `bev_pool_v2` 的定制版 `mmdet3d`。为保持 `/workspace/model/MapTrv2` 干净，在 `/tmp/maptrv2_ext_build` 中依次应用构建兼容补丁、PyTorch 2.10 API 适配和镜像内置 HIP 补丁后编译安装：

```bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1

rm -rf /tmp/maptrv2_ext_build
mkdir -p /tmp/maptrv2_ext_build/projects/mmdet3d_plugin/maptr/modules/ops
cp -a /workspace/model/MapTrv2/mmdetection3d /tmp/maptrv2_ext_build/
cp -a /workspace/model/MapTrv2/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn \
  /tmp/maptrv2_ext_build/projects/mmdet3d_plugin/maptr/modules/ops/

# 应用构建兼容补丁
python -m turbo_physai.optimizations.models.maptrv2.build /tmp/maptrv2_ext_build

# 适配 PyTorch 2.10 并编译 GeometricKernelAttention
GKA_DIR=/tmp/maptrv2_ext_build/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
sed -i 's/\.type()\.is_cuda()/.is_cuda()/g' \
  "$GKA_DIR/src/geometric_kernel_attn.h" \
  "$GKA_DIR/src/geometric_kernel_attn_cuda.cu"
sed -i 's/value\.type()/value.scalar_type()/g' \
  "$GKA_DIR/src/geometric_kernel_attn_cuda.cu"
cd "$GKA_DIR"
MAX_JOBS=8 python setup.py build install

# 叠加 HIP 补丁并编译 MapTRv2 版 mmdet3d
cd /tmp/maptrv2_ext_build/mmdetection3d
patch -p6 < /opt/autodrive/third_party/mmdetection3d/patches/0.17.2/0001-enable-tensorview-hip-device-functions.patch
patch -p6 < /opt/autodrive/third_party/mmdetection3d/patches/0.17.2/0002-use-hip-compatible-thread-blocks.patch
patch -p6 < /opt/autodrive/third_party/mmdetection3d/patches/0.17.2/0004-use-hip-compatible-voxel-thread-blocks.patch
MAX_JOBS=8 python -m pip install --no-deps --no-build-isolation --force-reinstall .

cd /workspace/model/MapTrv2 && rm -rf /tmp/maptrv2_ext_build
```

在 `/workspace/model/MapTrv2` 目录下验证扩展导入：

```bash
cd /workspace/model/MapTrv2
python - <<'PYTHON'
import torch, mmcv, mmdet, mmdet3d, shapely
import turbo_physai._C, GeometricKernelAttention
from apex.optimizers import FusedAdam
from mmdet3d.ops import bev_pool
from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
print('PyTorch:', torch.__version__, 'MMCV:', mmcv.__version__,
      'MMDet:', mmdet.__version__, 'MMDet3D:', mmdet3d.__version__)
PYTHON
```

## 5. 准备 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download)获取并接受许可后，按 [MapTR 官方数据准备说明](https://github.com/hustvl/MapTR/blob/maptrv2/docs/prepare_dataset.md)准备 nuScenes 数据、地图扩展和 MapTRv2 专用的 train/val info 文件。数据与验证地图标注路径如下：

```text
/workspace/model/MapTrv2/data/nuscenes -> /dataset/nuscenes
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── nuscenes_map_infos_temporal_train.pkl
└── nuscenes_map_infos_temporal_val.pkl
/workspace/anno_pkls/nuscenes_map_anns_val.json
```

链接数据目录并确认训练/验证标注文件存在：

```bash
cd /workspace/model/MapTrv2
[ -e data/nuscenes ] || [ -L data/nuscenes ] || ln -s /dataset/nuscenes data/nuscenes
ls -lh data/nuscenes/nuscenes_map_infos_temporal_{train,val}.pkl /workspace/anno_pkls/nuscenes_map_anns_val.json
```

TurboPhysAI 不下载、转换或重新分发 nuScenes 数据。

## 6. 准备预训练权重

R50 配置默认加载 `ckpts/resnet50-19c8e357.pth`：

```bash
cd /workspace/model/MapTrv2
mkdir -p ckpts
test -e ckpts/resnet50-19c8e357.pth || wget -O ckpts/resnet50-19c8e357.pth \
  https://download.pytorch.org/models/resnet50-19c8e357.pth
```

该权重为旧版 `.tar` 格式，PyTorch 2.6+ 需在启动命令中通过 `--set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 兼容加载。

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 MapTRv2 优化。MapTRv2 RuntimeConfig 包含训练所需的运行环境和 NUMA 配置。

### 7.1 生成并核对 OptimizationConfig

在目标容器的干净基线上生成与当前 PyTorch/MMCV 源码证据匹配的 `work_dirs/turbophysai/optimization.yaml`：

```bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/model/MapTrv2
mkdir -p work_dirs/turbophysai
RECIPE_YAML="$(python -c 'from pathlib import Path; import turbo_physai.optimizations.models.maptrv2 as m; print(Path(m.__file__).resolve().parent / "configs/recipe.yaml")')"

turbo-physai optimization generate \
  --recipe "$RECIPE_YAML" \
  --repo /workspace/model/MapTrv2 \
  --commit e03f097abef19e1ba3fed5f471a8d80fbfa0a064 \
  --output work_dirs/turbophysai/optimization.yaml \
  --force
turbo-physai optimization check work_dirs/turbophysai/optimization.yaml --repo /workspace/model/MapTrv2
```

### 7.2 快速验证（单机八卡）

运行 1 个 epoch（294 次迭代）并跳过验证评估；首次运行含 Inductor 冷编译（第 1 步耗时约 10–15 分钟属正常）：

```bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/model/MapTrv2
export SMOKE_OUTPUT="work_dirs/maptrv2_smoke_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$SMOKE_OUTPUT"

turbo-physai run \
  --model maptrv2 \
  --optimization-config work_dirs/turbophysai/optimization.yaml \
  --set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  --log-report \
  torchrun --nproc-per-node=8 --master-port=29501 tools/train.py \
    projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
    --launcher pytorch \
    --seed 0 \
    --no-validate \
    --work-dir "$SMOKE_OUTPUT" \
    --cfg-options \
      data.samples_per_gpu=12 \
      total_epochs=1 \
      runner.max_epochs=1 \
      log_config.interval=1 2>&1 | tee "$SMOKE_OUTPUT/train.log"
```

### 7.3 单机八卡训练

```bash
set -eo pipefail
source /opt/dtk/env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/model/MapTrv2
export TRAIN_OUTPUT="work_dirs/maptrv2_optimized_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$TRAIN_OUTPUT"

turbo-physai run \
  --model maptrv2 \
  --optimization-config work_dirs/turbophysai/optimization.yaml \
  --set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  --log-report \
  torchrun --nproc-per-node=8 --master-port=29513 tools/train.py \
    projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
    --launcher pytorch \
    --seed 0 \
    --work-dir "$TRAIN_OUTPUT" \
    --cfg-options \
      data.samples_per_gpu=12 \
      data.workers_per_gpu=4 \
      data.val.map_ann_file=/workspace/anno_pkls/nuscenes_map_anns_val.json \
      data.test.map_ann_file=/workspace/anno_pkls/nuscenes_map_anns_val.json \
      total_epochs=24 \
      runner.max_epochs=24 \
      evaluation.interval=24 \
      checkpoint_config.interval=2 \
      checkpoint_config.max_keep_ckpts=1 \
      log_config.interval=1 \
    2>&1 | tee "$TRAIN_OUTPUT/train.log"
```

`--nproc-per-node` 指定当前节点的训练进程数；如需指定通信网卡，可在 `torchrun` 前增加 `--set NCCL_SOCKET_IFNAME=<网卡名>`。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
