# PointPillars 应用说明

本文介绍 PointPillars 训练环境、模型源码及 nuScenes 数据准备，并使用 TurboPhysAI 应用训练优化。

## 1. 模型简介

[PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars) 是一种基于激光雷达点云的三维目标检测方法。本文使用 MMDetection3D 中基于 nuScenes 数据集的 PointPillars 配置。

## 2. 优化接入基线

TurboPhysAI 的 PointPillars 优化基于官方 [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) commit `fe25f7a51d36e3702f961e198894580d83c4387b` 接入，训练配置为 `configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py`。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备训练环境

按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用自动驾驶模型配套镜像创建并进入容器。

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2
export PYTHONDONTWRITEBYTECODE=1
mkdir -p /workspace/PointPillars_TurboPhysAI
```

## 4. 准备源码及配置文件

```bash
cd /workspace/PointPillars_TurboPhysAI
git clone https://github.com/open-mmlab/mmdetection3d.git \
  /workspace/PointPillars_TurboPhysAI/mmdetection3d
git -C /workspace/PointPillars_TurboPhysAI/mmdetection3d checkout --detach \
  fe25f7a51d36e3702f961e198894580d83c4387b
```

## 5. 准备 nuScenes 数据

从 [nuScenes 官方下载页](https://www.nuscenes.org/download) 按许可要求下载并解压 nuScenes v1.0 trainval 数据。将数据准备到容器中可访问的位置，本例使用 `/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars`：

```text
/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
├── nuscenes_infos_train.pkl
└── nuscenes_infos_val.pkl
```

### 5.1 生成训练和验证 PKL

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/PointPillars_TurboPhysAI/mmdetection3d${PYTHONPATH:+:$PYTHONPATH}

RUN_DIR=$(mktemp -d /workspace/PointPillars_TurboPhysAI/static_data/pkl_run_XXXXXX)
mkdir -p "$RUN_DIR/data"
ln -s /workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars "$RUN_DIR/data/nuscenes"
cd "$RUN_DIR"

python - <<'PY'
import sys
import types
from pathlib import Path

repo = '/workspace/PointPillars_TurboPhysAI/mmdetection3d'
sys.path.insert(0, repo)
tools = types.ModuleType('tools')
tools.__path__ = [f'{repo}/tools']
sys.modules['tools'] = tools

from tools.dataset_converters.nuscenes_converter import create_nuscenes_infos
from tools.dataset_converters.update_infos_to_v2 import update_pkl_infos

root = Path('./data/nuscenes')
for name in ('nuscenes_infos_train.pkl', 'nuscenes_infos_val.pkl'):
    path = root / name
    if path.exists() or path.is_symlink():
        raise SystemExit(f'文件已存在，不覆盖：{path}')

create_nuscenes_infos(str(root), 'nuscenes', version='v1.0-trainval', max_sweeps=10)
for split in ('train', 'val'):
    update_pkl_infos('nuscenes', out_dir=str(root), pkl_path=str(root / f'nuscenes_infos_{split}.pkl'))
PY
```

## 6. 通过 turbo-physai run 启动训练

启动前请根据实际环境调整以下变量或参数：

- `HIP_VISIBLE_DEVICES`：指定使用的 HCU 卡号；`--nproc_per_node` 设置为使用的卡数，两者数量应一致。本例使用 8 张卡。
- `train_dataloader.batch_size`：每张卡的 batch size，本例为 4；全局 batch size 等于每卡 batch size 乘以卡数。
- `train_dataloader.num_workers`：每张卡的数据加载进程数，本例为 10，可根据 CPU 资源调整。
- `DATA_ROOT`：nuScenes 数据目录，默认为 `/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars`。
- `POINTPILLARS_REPO_ROOT` 和 `POINTPILLARS_CONFIG`：分别指定 MMDetection3D 源码目录和训练配置文件，默认使用本例的源码和配置。
- `WORK_DIR` 和 `LOG_DIR`：分别指定训练输出目录和日志目录；`default_hooks.logger.interval` 控制训练日志输出间隔。

训练脚本示例如下：

```bash
#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2

DATA_ROOT=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}

cd "${POINTPILLARS_REPO_ROOT:-/workspace/PointPillars_TurboPhysAI/mmdetection3d}"
mkdir -p "${LOG_DIR:-/workspace/PointPillars_TurboPhysAI/logs}"

turbo-physai run \
    --model pointpillars \
    --set HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --set "PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    --log-report -- \
    torchrun --nproc_per_node 8 --standalone \
    tools/train.py "${POINTPILLARS_CONFIG:-configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR:-/workspace/PointPillars_TurboPhysAI/work_dirs/pointpillars_turbo_$(date +%Y%m%d_%H%M%S)}" \
    --cfg-options \
    train_dataloader.batch_size=4 \
    train_dataloader.num_workers=10 \
    default_hooks.logger.interval=1 \
    "train_dataloader.dataset.data_root=${DATA_ROOT}/" \
    "val_dataloader.dataset.data_root=${DATA_ROOT}/" \
    "test_dataloader.dataset.data_root=${DATA_ROOT}/" \
    "val_evaluator.data_root=${DATA_ROOT}/" \
    "val_evaluator.ann_file=${DATA_ROOT}/nuscenes_infos_val.pkl" \
    "test_evaluator.data_root=${DATA_ROOT}/" \
    "test_evaluator.ann_file=${DATA_ROOT}/nuscenes_infos_val.pkl" \
    "$@" 2>&1 | tee "${LOG_DIR:-/workspace/PointPillars_TurboPhysAI/logs}/pointpillars_$(date +%Y%m%d_%H%M%S)_$$.log"
```

## 7. 目录结构

容器中的主要目录如下：

```text
/workspace/PointPillars_TurboPhysAI/
├── mmdetection3d/                        # 官方模型源码
├── TurboPhysAI/
│   └── turbo_physai/optimizations/models/pointpillars/
│       ├── configs/
│       │   ├── optimization.yaml
│       │   ├── recipe.yaml
│       │   └── runtime.yaml
│       └── ...                           # 优化代码
└── static_data/
    └── nuscenes_pointpillars/
        ├── maps/
        ├── samples/
        ├── sweeps/
        ├── v1.0-trainval/
        ├── nuscenes_infos_train.pkl
        └── nuscenes_infos_val.pkl
```
