# OpenPI 应用说明

本文以 PI0.5 LIBERO 为例，说明如何在 HCU 环境中接入 TurboPhysAI 并启动训练。HCU 镜像和通用安装参见 [TurboPhysAI 安装指南](https://github.com/HYGON-AI/TurboPhysAI/blob/main/docs/zh/get_started/installation.md)。

## 模型简介

[OpenPI](https://github.com/Physical-Intelligence/openpi) 是 Physical Intelligence 开源的机器人学习代码库，提供 PI0、PI0-FAST、PI0.5 等视觉-语言-动作（VLA）模型及训练、微调和推理示例；模型根据视觉观测和语言指令生成机器人动作。本例使用 PyTorch 入口 `scripts/train_pytorch.py`，在单机 8 卡 HCU 上训练 `pi05_libero`。

## 优化接入基线

- OpenPI commit：`215abfb217dbac7d5f1273282331b9b1866c0479`
- TurboPhysAI OpenPI 集成：[openpi-TurboPhysAI-Optimization](https://github.com/Coulsonq/TurboPhysAI/tree/openpi-TurboPhysAI-Optimization)，位于 `turbo_physai/optimizations/models/openpi`
- 配置中的兼容性指纹对应上述 OpenPI commit；更换版本后需重新检查优化配置。

## 准备环境和源码

参考环境：Ubuntu 22.04、Python 3.10、DTK 26.04、PyTorch 2.10。使用与宿主机驱动匹配的 HCU 镜像，保留镜像自带的 Torch、TorchVision、FlashAttention 和 Triton。

容器挂载示例（镜像和宿主机目录按实际环境替换）：

```bash
mkdir -p "$PWD/openpi-workspace"

docker run -dit \
  --name openpi-turbophysai \
  --restart unless-stopped \
  --network host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=128G \
  --group-add video \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$PWD/openpi-workspace:/workspace/openpi_TurboPhysAI" \
  -w /workspace/openpi_TurboPhysAI \
  your-hcu-image:your-tag \
  bash -lc 'exec tail -f /dev/null'

docker exec -it openpi-turbophysai bash
```

容器内拉取固定版本源码，并构建安装 TurboPhysAI wheel：

```bash
cd /workspace/openpi_TurboPhysAI

git clone https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
git -C openpi submodule update --init --recursive

git clone --single-branch \
  --branch openpi-TurboPhysAI-Optimization \
  https://github.com/Coulsonq/TurboPhysAI.git TurboPhysAI

cd /workspace/openpi_TurboPhysAI/TurboPhysAI
python -c 'import torch; print("torch:", torch.__version__)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help
```

## 准备 LIBERO 数据集

下载 [LIBERO 数据集](https://huggingface.co/datasets/physical-intelligence/libero)。需要镜像时可设置 `HF_ENDPOINT`。

```bash
cd /workspace/openpi_TurboPhysAI
python -m pip install --no-deps --force-reinstall 'huggingface_hub==0.36.2'

# 可选：Hugging Face 镜像
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p static_data/.lerobot_root/physical-intelligence/libero
hf download physical-intelligence/libero \
  --repo-type dataset \
  --local-dir /workspace/openpi_TurboPhysAI/static_data/.lerobot_root/physical-intelligence/libero
```

计算 `norm_stats.json`，输出到 `static_data/assets/pi05_libero/physical-intelligence/libero/`：

```bash
cd /workspace/openpi_TurboPhysAI/static_data

export PYTHONPATH="/workspace/openpi_TurboPhysAI/openpi/src:/workspace/openpi_TurboPhysAI/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME=/workspace/openpi_TurboPhysAI/static_data/.lerobot_root
export JAX_PLATFORMS=cpu

python - <<'PY'
import datetime
import dataclasses
import runpy
import sys

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

import openpi.training.config as config
original_get_config = config.get_config
config.get_config = lambda name: dataclasses.replace(
    original_get_config(name), num_workers=0
)

script = "/workspace/openpi_TurboPhysAI/openpi/scripts/compute_norm_stats.py"
sys.argv = [script, "--config-name", "pi05_libero"]
runpy.run_path(script, run_name="__main__")
PY
```

## 准备 PI0.5 权重和 tokenizer

使用 [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) 下载官方 PI0.5 base checkpoint：

```bash
cd /workspace/openpi_TurboPhysAI
mkdir -p static_data/pi05_base

gcloud storage rsync \
  gs://openpi-assets/checkpoints/pi05_base \
  /workspace/openpi_TurboPhysAI/static_data/pi05_base \
  --recursive

test -f static_data/pi05_base/params/commit_success.txt \
  && echo DOWNLOAD_COMPLETE \
  || { echo 'Checkpoint download is incomplete'; exit 1; }
```

安装要求的 Transformers 版本，并应用 OpenPI 随附的 Transformers 补丁：

```bash
cd /workspace/openpi_TurboPhysAI

python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
assert transformers.__version__ == "4.53.2"
PY

TRANSFORMERS_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/transformers"
cp -a openpi/src/openpi/models_pytorch/transformers_replace/. "${TRANSFORMERS_DIR}/"
```

将 JAX checkpoint 转为 PyTorch 权重：

```bash
cd /workspace/openpi_TurboPhysAI/openpi

export PYTHONPATH="/workspace/openpi_TurboPhysAI/openpi/src:/workspace/openpi_TurboPhysAI/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

python - <<'PY'
import datetime
import runpy
import sys

if not hasattr(datetime, "UTC"):
    datetime.UTC = datetime.timezone.utc

script = "examples/convert_jax_model_to_pytorch.py"
sys.argv = [
    script,
    "--checkpoint_dir", "/workspace/openpi_TurboPhysAI/static_data/pi05_base",
    "--config_name", "pi05_libero",
    "--output_path", "/workspace/openpi_TurboPhysAI/static_data/pi05_base_pytorch",
    "--precision", "bfloat16",
]
runpy.run_path(script, run_name="__main__")
PY
```

下载 [PaliGemma tokenizer](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model)：

```bash
mkdir -p /workspace/openpi_TurboPhysAI/static_data/assets
curl -fL --retry 5 --retry-delay 2 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o /workspace/openpi_TurboPhysAI/static_data/assets/paligemma_tokenizer.model
```

## 检查配置并启动训练

检查随 OpenPI 集成提供的优化配置：

```bash
cd /workspace/openpi_TurboPhysAI/TurboPhysAI

export PYTHONPATH="/workspace/openpi_TurboPhysAI/TurboPhysAI:/workspace/openpi_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/openpi/compat:/workspace/openpi_TurboPhysAI/openpi/src:/workspace/openpi_TurboPhysAI/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

turbo-physai optimization check \
  turbo_physai/optimizations/models/openpi/configs/optimization.yaml \
  --repo /workspace/openpi_TurboPhysAI/openpi
```

按设备修改 `configs/runtime.yaml` 中的可见卡、数据集路径和 DTK/BLAS 库路径。启动脚本会设置运行目录、资源路径及输出目录：

```bash
cd /workspace/openpi_TurboPhysAI
source TurboPhysAI/turbo_physai/optimizations/models/openpi/scripts/prepare_tune_runtime.sh
set -o pipefail
```

按模型名启动（单机 8 卡，全局 batch size 256）：

```bash
turbo-physai run \
  --model openpi \
  --log-report \
  -- \
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "pi05_libero_8gpu_bs32_by_model_${STAMP}" \
    --batch-size 256 \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --pytorch_weight_path "${WEIGHT_PATH}" \
    --num_train_steps 10000 \
    --log-interval 1 \
    --save-interval 500 \
    --no-wandb-enabled \
  2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_model_${STAMP}.log"
```

显式指定优化配置和 Runtime 配置：

```bash
turbo-physai run \
  --optimization-config "${OPENPI_MODEL_DIR}/configs/optimization.yaml" \
  --runtime-config "${OPENPI_MODEL_DIR}/configs/runtime.yaml" \
  --log-report \
  -- \
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "pi05_libero_8gpu_bs32_by_config_${STAMP}" \
    --batch-size 256 \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --pytorch_weight_path "${WEIGHT_PATH}" \
    --num_train_steps 30000 \
    --log-interval 1 \
    --save-interval 2000 \
    --no-wandb-enabled \
    --seed 42 \
  2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_config_${STAMP}.log"
```

日志和检查点默认写入工作区的 `output/logs/` 与 `output/checkpoints/`。检查 `--log-report`，确认预期优化组状态为 `applied`。
