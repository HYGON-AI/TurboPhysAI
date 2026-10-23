# FlashOCC on HCU：TurboPhysAI 使用教程

本教程从自驾基础镜像开始，依次准备官方 FlashOCC 源码、TurboPhysAI、MMDetection3D、HCU 扩展、nuScenes 数据和预训练权重，最后启动训练。以下命令按同一个容器内的 shell 顺序执行。

## 1. 使用自驾基础镜像创建容器

请先在宿主机确认镜像 `autonomous-driving:torch2.10-dtk26.04` 已准备好，然后创建容器。数据集挂载仅适用于内部测试；这里不挂载宿主机数据，请按第 7 步自行下载。

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

## 2. 激活环境并设置工作目录

在容器内执行：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONDONTWRITEBYTECODE=1

export ROOT=/workspace/FlashOCC_TurboPhysAI
mkdir -p "$ROOT"
cd "$ROOT"
```

## 3. 拉取 FlashOCC 官方仓库

```bash
git clone https://github.com/Yzichen/FlashOCC.git "$ROOT/FlashOCC"
git -C "$ROOT/FlashOCC" checkout --detach \
  449dc7947d6549ccb72a99d6f6d8c02f166ec0dd
```

## 4. 拉取 TurboPhysAI 优化分支并安装

```bash
cd "$ROOT"
git clone --branch flashocc-optimization \
  https://github.com/Coulsonq/TurboPhysAI.git \
  "$ROOT/TurboPhysAI"

cd "$ROOT/TurboPhysAI"
python -c 'import torch; print(torch.__version__, torch.version.hip)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help

export PLUGIN="$ROOT/TurboPhysAI/turbo_physai/optimizations/models/flashocc"
```

## 5. 准备外置 MMDetection3D 依赖

```bash
export ROOT=/workspace/FlashOCC_TurboPhysAI
mkdir -p "$ROOT/runtime"
git clone --branch v1.0.0rc4 \
  https://github.com/open-mmlab/mmdetection3d.git \
  "$ROOT/runtime/mmdetection3d"
git -C "$ROOT/runtime/mmdetection3d" checkout --detach \
  c9541b0db89498fdea5cafd05b7b17f7b625b858

export PYTHONPATH="$ROOT/TurboPhysAI:$ROOT/runtime/mmdetection3d:$ROOT/FlashOCC${PYTHONPATH:+:$PYTHONPATH}"
```

## 6. 构建适配 HCU 的扩展

```bash
export ROOT=/workspace/FlashOCC_TurboPhysAI
export PLUGIN="$ROOT/TurboPhysAI/turbo_physai/optimizations/models/flashocc"
bash "$PLUGIN/scripts/build_flashocc_ops_external.sh"
```

## 7. 下载并准备 nuScenes 数据集

按照 [FlashOCC 官方安装文档 Step 3](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) 下载并准备 nuScenes 数据。将数据放到 `$ROOT/static_data/data/nuscenes`，目录至少包含：

```text
$ROOT/
├── FlashOCC/
├── runtime/
├── TurboPhysAI/
└── static_data/
    └── data/
        └── nuscenes/
            ├── v1.0-trainval/
            ├── sweeps/
            ├── samples/
            ├── gts/
            └── maps/
```

## 8. 生成 nuScenes 训练标注 PKL

```bash
export ROOT=/workspace/FlashOCC_TurboPhysAI
mkdir -p "$ROOT/static_data"
cd "$ROOT/static_data"

PYTHONPATH="$ROOT/TurboPhysAI:$ROOT/runtime/mmdetection3d:$ROOT/FlashOCC${PYTHONPATH:+:$PYTHONPATH}" \
python - <<'PY'
import runpy
import sys
import types

root = "/workspace/FlashOCC_TurboPhysAI/FlashOCC"
tools = types.ModuleType("tools")
tools.__path__ = [f"{root}/tools"]
sys.modules["tools"] = tools

runpy.run_path(
    f"{root}/tools/create_data_bevdet.py",
    run_name="__main__",
)
PY
```

## 9. 下载预训练权重

将 `bevdet-r50-cbgs.pth` 放入训练脚本使用的 `$ROOT/static_data/ckpts`：

```bash
mkdir -p "$ROOT/static_data/ckpts"
curl -fL --retry 5 --retry-delay 3 \
  -o "$ROOT/static_data/ckpts/bevdet-r50-cbgs.pth" \
  https://haihub-model-bj-1251001002.cos.ap-beijing.myqcloud.com/HyperDrive/ckpt/bevdet-r50-cbgs.pth
```

## 10. 启动训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1

bash /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/run_flashocc_turbo.sh
```
