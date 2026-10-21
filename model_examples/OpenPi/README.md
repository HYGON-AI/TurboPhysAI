# OpenPI 应用说明

本文说明如何在 Hygon HCU 环境中，基于 OpenPI 官方源码应用 TurboPhysAI 优化并启动 PI0.5 LIBERO 训练。示例容器内工作区为 `/workspace/openpi_TurboPhysAI`；若修改容器内挂载路径，请同步调整 `runtime.yaml` 和下文命令中的数据路径。

镜像获取和容器创建依赖具体 HCU/DTK 环境。TurboPhysAI OpenPI 集成不包含 HCU 驱动、DTK 或 PyTorch 二进制运行时，也不发布通用 Docker 镜像。本文假设已准备好兼容的 HCU 镜像和 OpenPI Python 依赖。若要实现从零构建镜像的开源复现，还需另外提供公开可访问的 Dockerfile、依赖文件和厂商基础镜像。

## 1. 模型简介

[OpenPI](https://github.com/Physical-Intelligence/openpi) 是 Physical Intelligence 发布的机器人模型代码库，包含 PI0、PI0-FAST 和 PI0.5 等视觉-语言-动作模型。本指南以 OpenPI PyTorch 训练入口 `scripts/train_pytorch.py` 和 `pi05_libero` 配置为例，在单机多卡 HCU 上进行 PI0.5 微调。

此处的 HCU 适配由 TurboPhysAI OpenPI 集成提供，不代表 OpenPI 上游对 HCU 的官方支持。OpenPI 上游硬件要求和功能边界请以[固定版本的官方说明](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)为准。

## 2. 优化接入基线

- OpenPI 官方源码固定在 commit `215abfb217dbac7d5f1273282331b9b1866c0479`。
- TurboPhysAI 的 OpenPI 集成位于 [`openpi-TurboPhysAI-Optimization`](https://github.com/Coulsonq/TurboPhysAI/tree/openpi-TurboPhysAI-Optimization) 分支，目录为 `turbo_physai/optimizations/models/openpi`。
- 上游合并进度见 [TurboPhysAI PR #22](https://github.com/HYGON-AI/TurboPhysAI/pull/22)。PR 合并后，可按上游版本和模型支持清单切换到官方分支。

`configs/optimization.yaml` 内的兼容性指纹对应上述 OpenPI commit。若切换 OpenPI 版本，需要重新检查或生成优化配置，不能假设原有优化仍可应用。

## 3. 准备 HCU 容器

基础镜像获取、TurboPhysAI 通用安装和容器创建方式参见[安装指南](https://github.com/HYGON-AI/TurboPhysAI/blob/main/docs/zh/get_started/installation.md)。本节给出 OpenPI 训练所需的 HCU 环境边界和一个容器挂载示例；OpenPI 模型源码、数据集、权重及训练输出不包含在 TurboPhysAI 镜像中。

参考环境为 Ubuntu 22.04、Python 3.10、DTK 26.04 和 PyTorch 2.10。请使用与你的宿主机驱动匹配且有权访问的 HCU 基础镜像，并保留镜像自带的 Torch、TorchVision、FlashAttention 和 Triton 版本；不要用通用 pip 命令覆盖厂商二进制组件。

下面是容器创建示例。镜像名需替换成实际镜像；宿主机目录挂载到容器内的固定工作区，以便数据、权重、日志和检查点能够持久保存。设备、驱动挂载、共享内存和安全参数按所在集群/厂商要求调整。`--privileged` 不作为默认选项，只有在厂商明确要求时才添加。

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

## 4. 准备源码并安装 TurboPhysAI

### 4.1 拉取 OpenPI 和 TurboPhysAI

以下命令在容器内执行：

```bash
cd /workspace/openpi_TurboPhysAI

git clone https://github.com/Physical-Intelligence/openpi.git openpi
git -C openpi checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
git -C openpi submodule update --init --recursive

git clone --single-branch \
  --branch openpi-TurboPhysAI-Optimization \
  https://github.com/Coulsonq/TurboPhysAI.git TurboPhysAI
```

### 4.2 构建并安装 TurboPhysAI

在包含兼容 Torch 和 HCU 编译环境的容器中构建 wheel。当前流程使用 wheel 安装，不使用 editable 安装：

```bash
cd /workspace/openpi_TurboPhysAI/TurboPhysAI

python -c 'import torch; print("torch:", torch.__version__)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help
```

`--no-deps` 用于避免安装 TurboPhysAI 时替换基础镜像中的二进制运行栈。构建失败时先核对 Python、Torch、编译器和 HCU 开发库的兼容性。

完成后，工作区主要结构如下：

```text
/workspace/openpi_TurboPhysAI/
├── openpi/          # 固定 commit 的 OpenPI 官方源码
├── TurboPhysAI/     # TurboPhysAI 源码及 OpenPI 模型集成
├── static_data/     # 数据集、权重和用户自行准备的 assets
└── output/          # 训练日志和检查点
```

## 5. 准备 LIBERO 数据集和归一化统计

下载的数据集应位于 LeRobot 约定的 `physical-intelligence/libero` 子目录下。`HF_ENDPOINT` 是可选镜像设置；只在你的网络环境需要且信任该镜像时启用。

```bash
cd /workspace/openpi_TurboPhysAI

python -m pip install --no-deps --force-reinstall 'huggingface_hub==0.36.2'

# 可选：使用 Hugging Face 镜像
# export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p static_data/.lerobot_root/physical-intelligence/libero
hf download physical-intelligence/libero \
  --repo-type dataset \
  --local-dir /workspace/openpi_TurboPhysAI/static_data/.lerobot_root/physical-intelligence/libero
```

若下载遇到 HTTP 429 等限流错误，保留现有目录并稍后重试同一命令，不要删除已下载的数据。请遵守 [LIBERO 数据集页面](https://huggingface.co/datasets/physical-intelligence/libero)中的使用条款。

### 计算 `norm_stats.json`

OpenPI 的统计脚本会读取 LeRobot 数据集，并将结果写入 assets 目录。下面示例将文件生成到 `static_data/assets/pi05_libero/physical-intelligence/libero/norm_stats.json`。Python 3.10 兼容补丁在导入 OpenPI 前设置；将 worker 数设为 0，避免子进程丢失此补丁。

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

## 6. 准备 PI0.5 权重和 tokenizer

### 6.1 下载并转换初始权重

OpenPI 上游 PI0.5 base checkpoint 位于 `gs://openpi-assets/checkpoints/pi05_base`。可以使用 [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) 下载；如果访问需要凭据，请按所在环境的授权流程配置。下载完成后先检查 `commit_success.txt`，不要转换不完整的 Orbax checkpoint。

```bash
cd /workspace/openpi_TurboPhysAI
mkdir -p static_data/pi05_base

gcloud storage rsync \
  gs://openpi-assets/checkpoints/pi05_base \
  /workspace/openpi_TurboPhysAI/static_data/pi05_base \
  --recursive

test -f static_data/pi05_base/params/commit_success.txt \
  && echo DOWNLOAD_COMPLETE \
  || { echo 'Checkpoint download is incomplete; do not convert it yet.'; exit 1; }
```

PyTorch PI0.5 需要 Transformers 4.53.2 和 OpenPI 仓库内的补丁。确认当前 Transformers 版本后，将补丁复制到当前 Python 环境的安装目录：

```bash
cd /workspace/openpi_TurboPhysAI

python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
assert transformers.__version__ == "4.53.2"
PY

TRANSFORMERS_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/transformers"
cp -a \
  openpi/src/openpi/models_pytorch/transformers_replace/. \
  "${TRANSFORMERS_DIR}/"
```

这会修改当前容器 Python 环境中的 Transformers 文件，建议在专用容器/环境中执行。转换命令如下；Python 3.10 下的 `datetime.UTC` 兼容补丁只影响此进程，不修改 OpenPI 源码：

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

转换成功后，`static_data/pi05_base_pytorch/` 下应包含 `model.safetensors` 和 `config.json`。

### 6.2 下载 PaliGemma tokenizer

tokenizer 是 OpenPI 运行资源，不是 TurboPhysAI 优化代码的一部分；单独下载到外部 `static_data/assets/`：

```bash
mkdir -p /workspace/openpi_TurboPhysAI/static_data/assets
curl -fL --retry 5 --retry-delay 2 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o /workspace/openpi_TurboPhysAI/static_data/assets/paligemma_tokenizer.model
```

训练前确认以下资源已就绪：

```text
static_data/
├── .lerobot_root/physical-intelligence/libero/       # LeRobot LIBERO 数据集
├── pi05_base/params/commit_success.txt               # 完整的 JAX checkpoint 标记
├── pi05_base_pytorch/model.safetensors               # 转换后的 PyTorch 权重
└── assets/
    ├── paligemma_tokenizer.model
    └── pi05_libero/physical-intelligence/libero/norm_stats.json
```

以上数据集、权重、tokenizer 和统计文件由使用者自行获取，不随 TurboPhysAI 优化包分发。

## 7. 检查并调整运行配置

OpenPI 集成配置位于：

```text
TurboPhysAI/turbo_physai/optimizations/models/openpi/
├── configs/recipe.yaml        # 优化配方
├── configs/optimization.yaml  # 针对固定 OpenPI commit 生成的配置
├── configs/runtime.yaml       # 环境变量和进程设置
└── scripts/                   # 运行时准备和启动脚本
```

固定 OpenPI commit 时可直接使用随包提供的 `optimization.yaml`。运行静态检查前，把兼容目录加入 `PYTHONPATH`，使 Python 3.10 的 `sitecustomize.py` 能在导入 OpenPI 前加载：

```bash
cd /workspace/openpi_TurboPhysAI/TurboPhysAI

export PYTHONPATH="/workspace/openpi_TurboPhysAI/TurboPhysAI:/workspace/openpi_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/openpi/compat:/workspace/openpi_TurboPhysAI/openpi/src:/workspace/openpi_TurboPhysAI/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

turbo-physai optimization check \
  turbo_physai/optimizations/models/openpi/configs/optimization.yaml \
  --repo /workspace/openpi_TurboPhysAI/openpi
```

首次运行前，按当前设备和路径检查 `configs/runtime.yaml`：

| 配置项 | 需要按环境确认的值 |
|---|---|
| `HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES` | 训练使用的 HCU 列表，应与 `torchrun --nproc_per_node` 一致 |
| `HF_LEROBOT_HOME` | 容器内 `static_data/.lerobot_root` 的绝对路径 |
| `LD_LIBRARY_PATH`、`ROCBLAS_TENSILE_LIBPATH` | 当前 DTK/BLAS 安装位置及匹配的 Tensile kernel 目录 |
| `OPENPI_DATA_HOME`、`HF_DATASETS_CACHE`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR` | 容器内可写且空间足够的缓存目录 |

`runtime.yaml` 中的 `HIP_BUFFER_EXTRA_SIZE=0` 是特定环境设置，不是通用 OpenPI 参数。请先确认当前 Hygon/DTK 版本的适用性，并验证训练正确性和稳定性后再决定是否保留。

## 8. 通过 `turbo-physai run` 启动训练

启动前先从工作区加载运行时准备脚本。该脚本推导源码、资源和输出路径，创建日志/检查点目录，并切换到 OpenPI 仓库；随后可以像 BEVFormer 示例一样直接调用 `turbo-physai run`。

```bash
cd /workspace/openpi_TurboPhysAI
source TurboPhysAI/turbo_physai/optimizations/models/openpi/scripts/prepare_tune_runtime.sh
set -o pipefail
```

### 8.1 使用模型名加载内置配置

`--model openpi` 自动加载集成包内的 OptimizationConfig 和 RuntimeConfig。下面示例为单机八卡、全局 batch size 256：

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

### 8.2 显式指定优化配置和 RuntimeConfig

需要选择或维护配置文件时，显式传入 `--optimization-config` 和 `--runtime-config`：

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

两种模式也分别封装在 `run_openpi_torchrun_tune_by_model.sh` 和 `run_openpi_torchrun_tune_by_config.sh` 中，可直接运行对应脚本。当前参考配置的训练步数和保存间隔如上；正式启动前请按硬件、训练目标和磁盘空间调整。卡数变化时同步修改 `runtime.yaml` 可见设备列表、`--nproc_per_node` 和全局 batch size，并保证 batch size 可被进程数整除。

日志和检查点默认保存到：

```text
output/logs/
output/checkpoints/
```

检查每个 rank 的 `--log-report` 执行结果，确认预期优化组为 `applied`，没有 `blocked` 或 `failed`。仅看到配置中的 `enabled: true` 不能证明优化已在训练进程中应用。

## 9. 许可与资源来源

- OpenPI 源码及模型许可：[固定版本 OpenPI](https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479)，请阅读其中的 `LICENSE`、`LICENSE_GEMMA.txt` 和模型资源条款。
- TurboPhysAI OpenPI 集成：[集成分支](https://github.com/Coulsonq/TurboPhysAI/tree/openpi-TurboPhysAI-Optimization)；上游进度见 [PR #22](https://github.com/HYGON-AI/TurboPhysAI/pull/22)。
- LIBERO 数据集：[Hugging Face 数据集页](https://huggingface.co/datasets/physical-intelligence/libero)。
- PaliGemma tokenizer：[Google Big Vision](https://storage.googleapis.com/big_vision/paligemma_tokenizer.model)。
- PI0.5 base checkpoint：`gs://openpi-assets/checkpoints/pi05_base`，由 OpenPI 官方仓库列出。
