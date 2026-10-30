# PointPillars 应用说明

本文说明如何在 HCU 环境中准备官方 PointPillars 训练所需的源码和 nuScenes 数据，并通过 TurboPhysAI 加载优化。示例在容器内使用 `mmcv2` 环境，不修改官方模型源码。

## 1. 模型简介

[PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars) 是一种基于激光雷达点云的三维目标检测方法。本文使用 MMDetection3D 中的 nuScenes PointPillars 配置。

## 2. 优化接入基线

本例使用官方 [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) commit `fe25f7a51d36e3702f961e198894580d83c4387b`，训练配置为 `configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py`。

PointPillars 的优化配置、运行配置及替换实现位于 [turbo_physai/optimizations/models/PointPillars](../../turbo_physai/optimizations/models/PointPillars/)。

## 3. 准备训练环境

本例使用 `autonomous-driving:torch2.10-dtk26.04` 和镜像内的 `mmcv2` Conda 环境。

以下命令在宿主机执行，设备、镜像名称及 `/opt/hyhal` 挂载按实际环境调整：

```bash
docker run -d -t \
  -v /opt/hyhal:/opt/hyhal:ro \
  --workdir /workspace/PointPillars_TurboPhysAI \
  --privileged \
  --shm-size=128G \
  --device=/dev/kfd \
  --device=/dev/dri/ \
  --device=/dev/mkfd \
  --network=host \
  --group-add video \
  --name PointPillars_TurboPhysAI \
  --entrypoint /bin/bash \
  autonomous-driving:torch2.10-dtk26.04 \
  -lc 'exec sleep infinity'

docker exec -it PointPillars_TurboPhysAI bash
```

后续命令均在容器内执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2
export PYTHONDONTWRITEBYTECODE=1
cd /workspace/PointPillars_TurboPhysAI
python -c 'import torch, mmcv, mmengine, mmdet; print("torch:", torch.__version__, "HIP:", torch.version.hip); print("mmcv:", mmcv.__version__, "mmengine:", mmengine.__version__, "mmdet:", mmdet.__version__)'
```

## 4. 准备源码并安装 TurboPhysAI

### 4.1 拉取官方模型源码

```bash
cd /workspace/PointPillars_TurboPhysAI
git clone https://github.com/open-mmlab/mmdetection3d.git \
  /workspace/PointPillars_TurboPhysAI/mmdetection3d
git -C /workspace/PointPillars_TurboPhysAI/mmdetection3d checkout --detach \
  fe25f7a51d36e3702f961e198894580d83c4387b
```

### 4.2 拉取并安装 TurboPhysAI

```bash
cd /workspace/PointPillars_TurboPhysAI
git clone --branch pointpillars-TurboPhysAI \
  https://github.com/Coulsonq/TurboPhysAI.git \
  /workspace/PointPillars_TurboPhysAI/TurboPhysAI
cd /workspace/PointPillars_TurboPhysAI/TurboPhysAI
python -c 'import torch; print(torch.__version__, torch.version.hip)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help
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

分支提供的启动脚本使用显式的 OptimizationConfig 和 RuntimeConfig，通过 `turbo-physai run` 启动官方 `tools/train.py`：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2
cd /workspace/PointPillars_TurboPhysAI
bash TurboPhysAI/turbo_physai/optimizations/models/PointPillars/scripts/run_pointpillars_turbo.sh
```

脚本当前默认使用物理卡 `0,1,2,3,4,5,6,7`、8 个进程、每卡 batch size 4 和每卡 10 个 data loader worker。设备不满足时需同时调整脚本中的 `HIP_VISIBLE_DEVICES` 和 `torchrun --nproc_per_node`；按实际环境设置可用通信网卡。RuntimeConfig 默认开启 NUMA，需要可用的 NUMA 工具和设备拓扑。

脚本支持通过环境变量设置数据和输出位置，例如：

```bash
DATA_ROOT=/path/to/nuscenes \
WORK_DIR=/workspace/PointPillars_TurboPhysAI/work_dirs/pointpillars_run \
LOG_DIR=/workspace/PointPillars_TurboPhysAI/logs \
bash /workspace/PointPillars_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/scripts/run_pointpillars_turbo.sh
```

`DATA_ROOT` 下需有本例数据结构及兼容的 PKL。默认日志保存在 `logs/`，训练输出保存在 `work_dirs/`，均位于官方模型仓库外。训练启动后应检查有效配置、优化报告和日志；`applied` 只表明替换入口应用，不证明所有优化均实际命中。

## 7. 目录结构

```text
/workspace/PointPillars_TurboPhysAI/
├── mmdetection3d/              # 官方模型源码
├── TurboPhysAI/                # TurboPhysAI 源码
│   ├── model_examples/PointPillars/README.md
│   └── turbo_physai/optimizations/models/PointPillars/
├── static_data/
│   ├── nuscenes_pointpillars/  # 数据和 PKL
```

## 8. 使用注意事项

- 优化组的目标和适用条件见 [PointPillars 接入说明](../../turbo_physai/optimizations/models/PointPillars/README.md)。部分替换有输入和硬件限制，可能回退到原实现。
- 当前版本完整训练精度、逐优化组对照、匹配基线吞吐及 loss 曲线仍需验证；不要把短时启动成功或配置加载作为完整验证结果。
- 如果 MMCV 或其 HCU 扩展与 PyTorch/DTK 或 MMDetection3D 不兼容，应先核对版本与二进制依赖，再进行适配。
