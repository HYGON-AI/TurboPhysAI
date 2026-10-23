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

python -m pip install --no-deps --force-reinstall 'huggingface_hub==0.36.2'

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

mkdir -p /workspace/openpi_TurboPhysAI/static_data/.lerobot_root/physical-intelligence/libero

hf download physical-intelligence/libero \
  --repo-type dataset \
  --local-dir /workspace/openpi_TurboPhysAI/static_data/.lerobot_root/physical-intelligence/libero
```

### 6.2 下载 PI0.5 base 权重

从官方 `gs://openpi-assets/checkpoints/pi05_base` 下载完整 JAX checkpoint。

```bash
cd /workspace/openpi_TurboPhysAI

python - <<'PY'
import http.client
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/"
DEST = Path("/workspace/openpi_TurboPhysAI/static_data/pi05_base")
DEST.mkdir(parents=True, exist_ok=True)
ROOT = DEST.resolve()

# 获取对象列表
objects = []
token = None
while True:
    query = {"prefix": PREFIX, "maxResults": "1000"}
    if token:
        query["pageToken"] = token
    url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urllib.parse.urlencode(query)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        page = json.load(response)
    objects.extend(page.get("items", []))
    token = page.get("nextPageToken")
    if not token:
        break

if not objects:
    raise RuntimeError(f"GCS 中没有找到对象：{PREFIX}")

def paths_for(obj):
    name = obj["name"]
    rel = name[len(PREFIX):]
    target = (ROOT / rel).resolve()
    target.relative_to(ROOT)  # 确认目标仍在下载目录内
    part = Path(str(target) + ".part")
    return target, part

def local_bytes():
    total = 0
    for obj in objects:
        expected = int(obj["size"])
        target, part = paths_for(obj)
        if target.exists() and target.stat().st_size == expected:
            total += expected
        elif part.exists():
            total += min(part.stat().st_size, expected)
    return total

def publish_part(part, target, expected):
    if target.exists():
        if target.stat().st_size == expected:
            return
        raise RuntimeError(f"正式文件已存在但大小不符，未覆盖：{target}")
    os.link(part, target)

total_bytes = sum(int(obj["size"]) for obj in objects)
initial_bytes = local_bytes()
started = time.monotonic()
print(f"{len(objects)} 个对象，共 {total_bytes / 1024**3:.2f} GiB；已有数据 {initial_bytes / 1024**3:.2f} GiB")

for index, obj in enumerate(objects, 1):
    expected = int(obj["size"])
    target, part = paths_for(obj)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        actual = target.stat().st_size
        if actual == expected:
            print(f"跳过 {index}/{len(objects)}：{obj['name']}（已完整）")
            continue
        raise RuntimeError(f"正式文件大小不符，为避免覆盖而停止：{target}")

    if part.exists() and part.stat().st_size > expected:
        raise RuntimeError(f".part 大于 GCS 对象大小，为避免覆盖而停止：{part}")

    if part.exists() and part.stat().st_size == expected:
        publish_part(part, target, expected)
        print(f"完成 {index}/{len(objects)}：{obj['name']}（已有完整 .part）")
        continue

    object_url = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/"
        f"{urllib.parse.quote(obj['name'], safe='')}"
    )
    params = {"alt": "media"}
    if obj.get("generation"):
        params["generation"] = obj["generation"]
    object_url += "?" + urllib.parse.urlencode(params)

    print(f"下载 {index}/{len(objects)}：{obj['name']}；"
          f"续传位置 {part.stat().st_size if part.exists() else 0} 字节")

    for attempt in range(1, 9):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(object_url, headers=headers)
        last_report = time.monotonic()
        bytes_since_report = 0

        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                status = getattr(response, "status", 200)
                if status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    match = re.match(r"bytes (\d+)-\d+/", content_range)
                    if not match or int(match.group(1)) != offset:
                        raise RuntimeError(f"服务器返回了不匹配的续传范围：{content_range}")
                elif offset:
                    raise RuntimeError(f"服务器未接受 Range 续传（HTTP {status}），未覆盖已有部分文件")

                mode = "ab" if offset else "wb"
                with open(part, mode, buffering=0) as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        bytes_since_report += len(chunk)

                        now = time.monotonic()
                        if bytes_since_report >= 64 * 1024**2 or now - last_report >= 10:
                            done = local_bytes()
                            elapsed = max(now - started, 0.001)
                            speed = max(done - initial_bytes, 0) / elapsed / 1024**2
                            remaining_mib = max(total_bytes - done, 0) / 1024**2
                            eta = (
                                f"{remaining_mib / (speed * 60):.1f} 分钟"
                                if speed >= 0.1 else "估算中"
                            )
                            print(f"总进度 {done / total_bytes * 100:.1f}%  "
                                  f"{speed:.1f} MiB/s  剩余约 {eta}")
                            last_report = now
                            bytes_since_report = 0

            actual = part.stat().st_size
            if actual != expected:
                raise EOFError(f"对象不完整：本地 {actual} 字节，预期 {expected} 字节")

            publish_part(part, target, expected)
            print(f"完成 {index}/{len(objects)}：{obj['name']}")
            break

        except urllib.error.HTTPError as error:
            if error.code not in (408, 429, 500, 502, 503, 504):
                raise
            failure = error
        except (TimeoutError, socket.timeout, urllib.error.URLError,
                http.client.HTTPException, ConnectionError, EOFError) as error:
            failure = error

        if attempt == 8:
            raise RuntimeError(
                f"重试 8 次仍失败：{obj['name']}；部分数据保留在 {part}"
            ) from failure

        wait = min(60, 2 ** (attempt - 1))
        offset = part.stat().st_size if part.exists() else 0
        print(f"读取失败：{failure}；{wait}s 后从 {offset} 字节处重试")
        time.sleep(wait)

print(f"\n下载完成：{ROOT}")
PY
```

### 6.3 应用 Transformers 补丁

将 OpenPI 提供的 Transformers 补丁复制到当前 Python 环境，并安装兼容版本的 huggingface-hub。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

TRANSFORMERS_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/transformers"

cp -a \
  /workspace/openpi_TurboPhysAI/openpi/src/openpi/models_pytorch/transformers_replace/. \
  "${TRANSFORMERS_DIR}/"

python -m pip install --no-cache-dir --no-deps --force-reinstall \
  'huggingface-hub>=0.30.0,<1.0'
```

### 6.4 转换为 PyTorch 格式

将下载完成的 JAX 权重转换为 PyTorch 格式，结果保存到 `static_data/pi05_base_pytorch`。

```bash
cd /workspace/openpi_TurboPhysAI/openpi

test -f /workspace/openpi_TurboPhysAI/static_data/pi05_base/params/commit_success.txt \
  && echo DOWNLOAD_COMPLETE \
  || echo "下载尚未完成，先不要转换"

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

```bash
mkdir -p /workspace/openpi_TurboPhysAI/static_data/assets
curl -fL --retry 5 --retry-delay 2 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o /workspace/openpi_TurboPhysAI/static_data/assets/paligemma_tokenizer.model
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

### 7.1 显式指定配置

```bash
cd /workspace/openpi_TurboPhysAI
source TurboPhysAI/turbo_physai/optimizations/models/openpi/scripts/prepare_tune_runtime.sh

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
  2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_config_${STAMP}.log"
```

### 7.2 按模型名加载配置

```bash
cd /workspace/openpi_TurboPhysAI
source TurboPhysAI/turbo_physai/optimizations/models/openpi/scripts/prepare_tune_runtime.sh

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
│       ├── paligemma_tokenizer.model
│       └── pi05_libero/physical-intelligence/libero/norm_stats.json
└── output/                                  # 训练输出
    ├── logs/
    └── checkpoints/
```
