# OpenPI 应用说明

本文说明如何在 HCU 环境中，直接使用 VLA 基础镜像，在容器内安装依赖、准备资源并应用 TurboPhysAI 优化，无需 Dockerfile。通用环境说明参见[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)。

## 1. 模型简介

[OpenPI](https://github.com/Physical-Intelligence/openpi) 是 Physical Intelligence 开源的机器人学习代码库，提供 PI0、PI0-FAST 和 PI0.5 等视觉-语言-动作（VLA）模型，根据视觉观测和语言指令生成机器人动作，用于机器人操作任务。

## 2. 优化接入基线

TurboPhysAI 的 OpenPI 优化基于官方仓库 commit `215abfb217dbac7d5f1273282331b9b1866c0479` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备训练环境

以下命令在宿主机执行:

### 3.1 拉取基础镜像

```bash
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10
```

### 3.2 创建并进入容器

```bash
docker run -dit \
  --restart unless-stopped \
  --network=host \
  --name=openpi-train-TurboPhysAI \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=128G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -w /workspace/openpi_TurboPhysAI \
  harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10 \
  bash -lc 'exec tail -f /dev/null'

docker exec -it openpi-train-TurboPhysAI bash
```

## 4. 准备模型源码并安装 TurboPhysAI

下文命令均在容器内执行。基础镜像保留原有 Torch、TorchVision、FlashAttention 和 Triton，Python 依赖与独立 BLAS 在后续步骤安装。

### 4.1 准备官方 OpenPI 源码

```bash
cd /workspace/openpi_TurboPhysAI
git clone https://github.com/Physical-Intelligence/openpi.git \
  /workspace/openpi_TurboPhysAI/openpi
git -C /workspace/openpi_TurboPhysAI/openpi checkout --detach \
  215abfb217dbac7d5f1273282331b9b1866c0479
git -C /workspace/openpi_TurboPhysAI/openpi submodule update --init --recursive
```

### 4.2 拉取 TurboPhysAI 源码

```bash
cd /workspace/openpi_TurboPhysAI
git clone --branch main \
  https://github.com/HYGON-AI/TurboPhysAI.git TurboPhysAI
```

### 4.3 编译并安装 TurboPhysAI

```bash
cd /workspace/openpi_TurboPhysAI/TurboPhysAI

python -c 'import torch; print(torch.__version__)'
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps --force-reinstall dist/turbo_physai-*.whl
turbo-physai --help
```

## 5. 安装 Python 依赖

普通依赖通过 [requirements.txt](./requirements.txt) 安装。

```bash
cd /workspace/openpi_TurboPhysAI/TurboPhysAI

python -m pip install --no-cache-dir \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r model_examples/OpenPi/requirements.txt
```

LeRobot 和 gym-aloha 使用 `--no-deps` 安装，避免其依赖要求替换基础镜像的 Torch 栈。

```bash
python -m pip install --no-cache-dir --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  gym-aloha==0.1.1 \
  'lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
```

## 6. 准备必要资源

### 6.1 下载 LIBERO 数据集

从 [Hugging Face](https://huggingface.co/datasets/physical-intelligence/libero) 或使用以下命令下载 LIBERO 数据集。

```bash
cd /workspace/openpi_TurboPhysAI

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p /workspace/openpi_TurboPhysAI/static_data/.lerobot_root/physical-intelligence/libero

hf download physical-intelligence/libero \
  --repo-type dataset \
  --local-dir /workspace/openpi_TurboPhysAI/static_data/.lerobot_root/physical-intelligence/libero
```

### 6.2 下载 PI0.5 base 权重

先按 [Google Cloud CLI 文档](https://cloud.google.com/storage/docs/gsutil_install) 安装 `gsutil`，再下载官方 JAX checkpoint。命令成功结束后再进行格式转换；重新运行可继续同步未完成的文件。`commit_success.txt` 只是 checkpoint 标记，不能单独证明下载完整。

```bash
mkdir -p /workspace/openpi_TurboPhysAI/static_data/pi05_base

gsutil -m rsync -r \
  gs://openpi-assets/checkpoints/pi05_base \
  /workspace/openpi_TurboPhysAI/static_data/pi05_base
```

### 6.3 应用 Transformers 补丁

将 OpenPI 提供的 Transformers 补丁复制到当前 Python 环境。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

TRANSFORMERS_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/transformers"

cp -a \
  /workspace/openpi_TurboPhysAI/openpi/src/openpi/models_pytorch/transformers_replace/. \
  "${TRANSFORMERS_DIR}/"
```

### 6.4 转换为 PyTorch 格式

将下载完成的 JAX 权重转换为 PyTorch 格式，结果保存到 `static_data/pi05_base_pytorch`。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

test -f /workspace/openpi_TurboPhysAI/static_data/pi05_base/params/commit_success.txt \
  || { echo "未找到 checkpoint 标记，请先完成下载"; exit 1; }

export PYTHONPATH="/workspace/openpi_TurboPhysAI/openpi/src:/workspace/openpi_TurboPhysAI/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu

python - <<'PY'
import datetime
import runpy
import sys

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

### 6.5 准备归一化统计

使用本例的 LIBERO 数据时，可直接使用附件中的 [norm_stats.json](./norm_stats.json)：

```bash
mkdir -p /workspace/openpi_TurboPhysAI/static_data/assets/pi05_libero/physical-intelligence/libero
cp /workspace/openpi_TurboPhysAI/TurboPhysAI/model_examples/OpenPi/norm_stats.json \
  /workspace/openpi_TurboPhysAI/static_data/assets/pi05_libero/physical-intelligence/libero/norm_stats.json
```

需要重新计算，或使用其他数据集时，运行以下命令；更换数据集需同步调整训练配置名称。结果保存到 `static_data/assets/pi05_libero/physical-intelligence/libero/norm_stats.json`。

```bash
python -m pip install --no-deps multiprocess
python -m pip install --no-deps attr

cd /workspace/openpi_TurboPhysAI/static_data

export PYTHONPATH="/workspace/openpi_TurboPhysAI/openpi/src:/workspace/openpi_TurboPhysAI/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_LEROBOT_HOME="/workspace/openpi_TurboPhysAI/static_data/.lerobot_root"
export JAX_PLATFORMS=cpu

python - <<'PY'
import datetime
import dataclasses
import runpy
import sys

# Python 3.10 兼容：必须在导入 OpenPI 模块前设置
datetime.UTC = datetime.timezone.utc

import openpi.training.config as config

# 关闭统计脚本的数据加载 worker，避免子进程丢失上述兼容补丁
original_get_config = config.get_config
config.get_config = lambda name: dataclasses.replace(
    original_get_config(name), num_workers=0
)

script = "/workspace/openpi_TurboPhysAI/openpi/scripts/compute_norm_stats.py"
sys.argv = [script, "--config-name", "pi05_libero"]
runpy.run_path(script, run_name="__main__")
PY
```

### 6.6 下载 PaliGemma tokenizer

直接下载到运行时 `OPENPI_DATA_HOME` 对应的缓存位置；以下路径与默认 `runtime.yaml` 一致。若修改了 `OPENPI_DATA_HOME`，请同步修改下载位置。`/dev/shm` 是临时存储，重启或清理后需重新下载。

```bash
mkdir -p /dev/shm/openpi_8gpu_cache/openpi_data/big_vision
curl -fL --retry 5 --retry-delay 2 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o /dev/shm/openpi_8gpu_cache/openpi_data/big_vision/paligemma_tokenizer.model
```

### 6.7 安装独立 BLAS 优化包

在容器内下载并解压 BLAS 包，创建与 `runtime.yaml` 对应的库路径。以下命令用于首次安装，目标软链接应尚不存在。

```bash
(
  set -Eeuo pipefail

  mkdir -p /opt/openpi/dtk-26.04-blas-raw

  curl -fsSL --retry 3 --retry-delay 2 \
    http://42.228.13.241:18000/dtk-archived/nightly/dtk-26.04/0921/DTK-26.04-blas-nightly-0921-cut-ubuntu20.04-x86_64.tar.gz \
    -o /tmp/dtk-blas.tar.gz

  tar -xzf /tmp/dtk-blas.tar.gz \
    -C /opt/openpi/dtk-26.04-blas-raw

  BLAS_LIB_DIR="$(find /opt/openpi/dtk-26.04-blas-raw \
    -type f -name 'librocblas.so*' -printf '%h\n' | head -n 1)"

  test -n "$BLAS_LIB_DIR"
  mkdir -p /opt/openpi/dtk-26.04-blas
  ln -s "$BLAS_LIB_DIR" /opt/openpi/dtk-26.04-blas/lib
  test -e /opt/openpi/dtk-26.04-blas/lib/librocblas.so
  echo "BLAS 安装并验证成功：$BLAS_LIB_DIR"
)
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 加载随组件交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)。按设备调整 RuntimeConfig 的可见卡、数据集和库路径。`prepare_tune_runtime.sh` 设置资源和输出路径，并切换到 OpenPI 源码目录。

可以直接执行以下任一脚本：

下面的 `OPENPI_NET_IFACE` 是通信网卡名称示例，请按本节点实际情况修改；多机训练时各节点应选择相互连通的网卡。单机训练如果不需要指定网卡，可删除 `--set NCCL_SOCKET_IFNAME` 参数。

当前分支仍包含 `openpi.network`，启动命令暂时通过 `--disable-group openpi.network` 关闭该组，避免它覆盖用户指定的网卡。后续同步移除该优化组的代码后，应同时删除这条 `--disable-group` 参数。

### 7.1 显式指定配置

```bash
cd /workspace/openpi_TurboPhysAI
source TurboPhysAI/turbo_physai/optimizations/models/openpi/scripts/prepare_tune_runtime.sh
export OPENPI_NET_IFACE=ens65f0np0  # 修改为本节点实际通信网卡

turbo-physai run \
  --optimization-config "${OPENPI_MODEL_DIR}/configs/optimization.yaml" \
  --runtime-config "${OPENPI_MODEL_DIR}/configs/runtime.yaml" \
  --set "NCCL_SOCKET_IFNAME=${OPENPI_NET_IFACE}" \
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
  2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_config_${STAMP}.log"
```

### 7.2 按模型名加载配置

```bash
cd /workspace/openpi_TurboPhysAI
source TurboPhysAI/turbo_physai/optimizations/models/openpi/scripts/prepare_tune_runtime.sh
export OPENPI_NET_IFACE=ens65f0np0  # 修改为本节点实际通信网卡

turbo-physai run \
  --model openpi \
  --disable-group openpi.network \
  --set "NCCL_SOCKET_IFNAME=${OPENPI_NET_IFACE}" \
  --log-report \
  -- \
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "pi05_libero_8gpu_bs32_by_model_${STAMP}" \
    --batch-size 256 \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --pytorch_weight_path "${WEIGHT_PATH}" \
    --num_train_steps 30000 \
    --log-interval 1 \
    --save-interval 2000 \
    --no-wandb-enabled \
  2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_model_${STAMP}.log"
```

训练日志和检查点默认保存到 `/workspace/openpi_TurboPhysAI/output/`。优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。

## 8. 目录结构

完成上述流程后，容器内主要目录如下：

```text
/workspace/openpi_TurboPhysAI/
├── openpi/                                  # OpenPI 官方源码
├── TurboPhysAI/                             # TurboPhysAI 源码
│   ├── model_examples/OpenPi/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── norm_stats.json                  # 本例提供的 LIBERO 统计文件
│   └── turbo_physai/optimizations/models/openpi/
├── static_data/                             # 数据集、权重和运行资源
│   ├── .lerobot_root/physical-intelligence/libero/
│   │   ├── data/
│   │   └── meta/
│   ├── pi05_base/params/                    # 原始 JAX 权重
│   ├── pi05_base_pytorch/                   # 转换后的 PyTorch 权重
│   │   ├── config.json
│   │   └── model.safetensors
│   └── assets/
│       └── pi05_libero/physical-intelligence/libero/norm_stats.json
└── output/                                  # 训练输出
    ├── logs/
    └── checkpoints/
```

tokenizer 位于工作目录之外的 OpenPI 缓存目录：

```text
/dev/shm/openpi_8gpu_cache/openpi_data/
└── big_vision/
    └── paligemma_tokenizer.model             # PaliGemma tokenizer 缓存
```

`/dev/shm` 是临时存储，重启或清理后可能需要重新下载 tokenizer。
