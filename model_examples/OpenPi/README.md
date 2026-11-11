# OpenPI 应用说明

本文介绍 OpenPI 依赖安装、资源准备及训练启动。

## 1. 模型简介

[OpenPI](https://github.com/Physical-Intelligence/openpi) 是 Physical Intelligence 开源的机器人学习代码库，提供 PI0、PI0-FAST 和 PI0.5 等视觉-语言-动作（VLA）模型，根据视觉观测和语言指令生成机器人动作，用于机器人操作任务。

## 2. 优化接入基线

TurboPhysAI 的 OpenPI 优化基于官方仓库 commit `215abfb217dbac7d5f1273282331b9b1866c0479` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备训练环境

按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 VLA、DiT 和世界模型配套镜像创建容器并安装 TurboPhysAI。下文命令均在容器内执行。

## 4. 准备模型源码及示例文件

### 4.1 准备官方 OpenPI 源码

```bash
mkdir -p /workspace/openpi_TurboPhysAI
cd /workspace/openpi_TurboPhysAI
git clone https://github.com/Physical-Intelligence/openpi.git \
  /workspace/openpi_TurboPhysAI/openpi
git -C /workspace/openpi_TurboPhysAI/openpi checkout --detach \
  215abfb217dbac7d5f1273282331b9b1866c0479
git -C /workspace/openpi_TurboPhysAI/openpi submodule update --init --recursive
```

以可编辑模式安装 OpenPI 和 OpenPI Client，供训练时导入。当前镜像使用 Python 3.10，因此使用 `--ignore-requires-python`；运行时仍需对应的兼容处理。`--no-deps` 避免替换镜像中的依赖。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

/usr/bin/python3 -m pip install --ignore-requires-python --no-deps \
  -e . \
  -e packages/openpi-client
```

### 4.2 依赖清单

克隆 TurboPhysAI 仓库仅用于获取 `model_examples/OpenPi` 下的依赖清单、归一化统计文件和权重下载脚本。

后续使用 `/usr/bin/python3` 安装依赖和启动分布式训练，并通过 `turbo-physai run` 应用优化；请确保 `turbo-physai` 和 `hf` 也属于同一个 Python 环境。

```bash
cd /workspace/openpi_TurboPhysAI
git clone --branch main \
  https://github.com/HYGON-AI/TurboPhysAI.git TurboPhysAI
```

## 5. 安装 OpenPI 依赖

普通依赖通过 [requirements.txt](./requirements.txt) 安装。

```bash
cd /workspace/openpi_TurboPhysAI/TurboPhysAI

/usr/bin/python3 -m pip install --no-cache-dir \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r model_examples/OpenPi/requirements.txt
```

LeRobot 和 gym-aloha 使用 `--no-deps` 安装，避免其依赖要求替换基础镜像的 Torch 栈。

```bash
/usr/bin/python3 -m pip install --no-cache-dir --no-deps \
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

使用 Python 脚本下载官方 JAX checkpoint，`--output-dir` 可按实际情况修改：

```bash
/usr/bin/python3 /workspace/openpi_TurboPhysAI/TurboPhysAI/model_examples/OpenPi/download_pi05_base.py \
  --output-dir /workspace/openpi_TurboPhysAI/static_data/pi05_base
```

下载成功后再进行格式转换。中断后可重新运行下载命令。若修改下载目录，请同步修改 6.4 节中的 checkpoint 检查路径和 `--checkpoint_dir`，两者均指向相同的下载目录。

### 6.3 应用 Transformers 补丁

将 OpenPI 提供的 Transformers 补丁复制到当前 Python 环境。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

TRANSFORMERS_DIR="$(/usr/bin/python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/transformers"

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

export JAX_PLATFORMS=cpu

/usr/bin/python3 - <<'PY'
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
/usr/bin/python3 -m pip install --no-deps multiprocess
/usr/bin/python3 -m pip install --no-deps attr

cd /workspace/openpi_TurboPhysAI/static_data

export HF_LEROBOT_HOME="/workspace/openpi_TurboPhysAI/static_data/.lerobot_root"
export JAX_PLATFORMS=cpu

/usr/bin/python3 - <<'PY'
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

将 tokenizer 下载到 `static_data/assets` 下的 OpenPI 缓存目录。下方训练命令通过 `--set` 指定相同的 `OPENPI_DATA_HOME`。

```bash
mkdir -p /workspace/openpi_TurboPhysAI/static_data/assets/big_vision
curl -fL --retry 5 --retry-delay 2 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o /workspace/openpi_TurboPhysAI/static_data/assets/big_vision/paligemma_tokenizer.model
```

## 7. 通过 turbo-physai run 启动训练
启动训练前，先创建输出目录：
```bash
mkdir -p \
  /workspace/openpi_TurboPhysAI/output/checkpoints \
  /workspace/openpi_TurboPhysAI/output/logs
```

`turbo-physai run` 加载随组件交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)。

请按实际设备和数据位置调整路径、可见卡及通信网卡；示例使用八卡，每卡 batch size 为 32。

启动命令从已安装的 TurboPhysAI 包获取 OpenPI 优化目录，并通过 `PYTHONPATH` 加载其中的 `sitecustomize.py`，确保训练进程及数据加载子进程应用兼容处理。`OPENPI_DATA_HOME` 和 `HF_LEROBOT_HOME` 分别指定资源缓存及数据集目录，请按实际情况调整。

`ens65f0np0` 仅为通信网卡名称示例，请按本节点实际情况修改；多机训练时各节点应选择相互连通的网卡。单机训练如果不需要指定网卡，可删除 `--set NCCL_SOCKET_IFNAME` 参数。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

STAMP="$(date +%Y%m%d_%H%M%S)"
OPENPI_TURBO_DIR="$(/usr/bin/python3 -c \
  'import os, turbo_physai.optimizations.models.openpi as m; print(os.path.dirname(m.__file__))')"

turbo-physai run \
  --model openpi \
  --set "PYTHONPATH=${OPENPI_TURBO_DIR}" \
  --set OPENPI_DATA_HOME=/workspace/openpi_TurboPhysAI/static_data/assets \
  --set HF_LEROBOT_HOME=/workspace/openpi_TurboPhysAI/static_data/.lerobot_root \
  --set NCCL_SOCKET_IFNAME=ens65f0np0 \
  --set HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --log-report \
  -- \
  /usr/bin/python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "pi05_libero_8gpu_bs32_by_config_${STAMP}" \
    --batch-size 256 \
    --assets-base-dir /workspace/openpi_TurboPhysAI/static_data/assets \
    --checkpoint-base-dir /workspace/openpi_TurboPhysAI/output/checkpoints \
    --pytorch_weight_path /workspace/openpi_TurboPhysAI/static_data/pi05_base_pytorch \
    --num_train_steps 30000 \
    --log-interval 100 \
    --save-interval 1000 \
    --no-wandb-enabled \
  2>&1 | tee -a "/workspace/openpi_TurboPhysAI/output/logs/pi05_libero_8gpu_bs32_by_config_${STAMP}.log"
```

训练日志和检查点默认保存到 `/workspace/openpi_TurboPhysAI/output/`。优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。

## 8. 目录结构

完成上述流程后，容器内主要目录如下。训练使用的 TurboPhysAI Python 包和配置位于 Python 环境的安装目录，不在此目录树中：

```text
/workspace/openpi_TurboPhysAI/
├── openpi/                                  # OpenPI 官方源码
├── TurboPhysAI/                             # 示例文件所在的源码仓库
│   └── model_examples/OpenPi/
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   ├── norm_stats.json                  # 本例提供的 LIBERO 统计文件
│   │   └── download_pi05_base.py           # PI0.5 base 权重下载脚本
├── static_data/                             # 数据集、权重和运行资源
│   ├── .lerobot_root/physical-intelligence/libero/
│   │   ├── data/
│   │   └── meta/
│   ├── pi05_base/params/                    # 原始 JAX 权重
│   ├── pi05_base_pytorch/                   # 转换后的 PyTorch 权重
│   │   ├── config.json
│   │   └── model.safetensors
│   └── assets/
│       ├── pi05_libero/physical-intelligence/libero/norm_stats.json
│       └── big_vision/paligemma_tokenizer.model
└── output/                                  # 训练输出
    ├── logs/
    └── checkpoints/
```

已安装的 TurboPhysAI Python 包示例结构如下：

```text
/usr/local/lib/python3.10/dist-packages/
└── turbo_physai/
    └── optimizations/models/openpi/
        ├── sitecustomize.py
        ├── catalog.py
        ├── ...                            # 其他优化实现
        └── configs/
            ├── .optimization.yaml.generation.json
            ├── optimization.yaml
            ├── recipe.yaml
            └── runtime.yaml
```

