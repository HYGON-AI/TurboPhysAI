# LingBot-VA 应用说明

本文说明如何在产品镜像中，基于官方 LingBot-VA 仓库应用 TurboPhysAI 的 HCU 优化。基于官方。

## 1. 模型简介

[LingBot-VA](https://github.com/Robbyant/lingbot-va) 联合建模视频世界状态和机器人动作。本接入面向官方 LIBERO 后训练配置。

## 2. 优化接入基线

TurboPhysAI 的 LingBot-VA 优化基于官方仓库 commit `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 下载并启动基础镜像

```bash
export BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10-20260929
docker pull "$BASE_IMAGE"

export WORKSPACE="$HOME/work/lingbot-va-workspace"
mkdir -p "$WORKSPACE"
docker run -dit \
  --name lingbot-va \
  --network=host \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=512G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$WORKSPACE":/workspace \
  -w /workspace \
  "$BASE_IMAGE" \
  sleep infinity
docker exec -it lingbot-va bash
```

## 4. 准备模型源码

```bash
cd /workspace
git clone https://github.com/HYGON-AI/TurboPhysAI.git
mkdir -p model
git clone https://github.com/Robbyant/lingbot-va.git model/lingbot-va
cd model/lingbot-va
git checkout --detach 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
git status --short
```

## 5. 安装模型源码依赖

使用 Python 3.10、DTK 26.04 的 HCU 环境。镜像中的 PyTorch、Triton 和 DTK 版 FlashAttention 必须保留，不能安装上游 CUDA 依赖覆盖它们。

```bash
cd /workspace/TurboPhysAI
MAX_JOBS=8 python3 setup.py bdist_wheel
python3 -m pip install --no-deps dist/turbo_physai-*.whl
python3 -m pip install --no-deps -r model_examples/LingBot-VA/requirements.txt
python3 -m pip check
```

已验证的软件版本见本目录的 `requirements.txt`；HCU 基础栈为 PyTorch `2.10.0+das.opt1.dtk2604.2608131652.g28498c`、Triton `3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1`、FlashAttention `2.8.4+dtk2604.torch2100.2609231125.g11f180`。

## 6. 准备 LIBERO 数据

下载官方 LIBERO-Long LeRobot 数据集：

```bash
mkdir -p /workspace/datasets/libero-long-lerobot
huggingface-cli download --repo-type dataset robbyant/libero-long-lerobot --local-dir /workspace/datasets/libero-long-lerobot
```

配置训练数据路径并关闭 WandB：

```bash
export DATA_ROOT=/workspace/datasets/libero-long-lerobot
python3 - <<'PYCODE'
from pathlib import Path
p = Path('/workspace/model/lingbot-va/wan_va/configs/va_libero_train_cfg.py')
s = p.read_text()
s = s.replace("'/path/to/your/dataset'", repr('/workspace/datasets/libero-long-lerobot'))
s = s.replace('va_libero_train_cfg.enable_wandb = True', 'va_libero_train_cfg.enable_wandb = False')
p.write_text(s)
PYCODE
test -f "$DATA_ROOT/empty_emb.pt"
```

数据目录应包含 `empty_emb.pt`、LeRobot 元数据、视频及对应的 `latents/`。

## 7. 准备预训练权重

下载 LingBot-VA base 权重：

```bash
mkdir -p /workspace/models/lingbot-va-base
huggingface-cli download robbyant/lingbot-va-base --local-dir /workspace/models/lingbot-va-base
```

训练必须使用 `attn_mode=flex`。设置模型路径并更新模型配置：

```bash
export MODEL_ROOT=/workspace/models/lingbot-va-base
python3 - <<'PYCODE'
import json
from pathlib import Path
model = Path('/workspace/models/lingbot-va-base')
p = model / 'transformer/config.json'
config = json.loads(p.read_text())
config['attn_mode'] = 'flex'
p.write_text(json.dumps(config, indent=2) + chr(10))

p = Path('/workspace/model/lingbot-va/wan_va/configs/va_libero_cfg.py')
s = p.read_text().replace('"/path/to/pretrained/model"', repr(str(model)))
p.write_text(s)
PYCODE
test -f "$MODEL_ROOT/transformer/config.json"
```

## 8. 通过 turbo-physai run 启动训练

`--model lingbot_va` 自动加载随包交付的九组 OptimizationConfig 和 NUMA RuntimeConfig。启动时使用 `turbo-physai run`，以便 Group 配置应用到训练进程及编译 worker。

### 8.1 单机八卡训练

```bash
cd /workspace/TurboPhysAI
export LINGBOT_VA_ROOT=/workspace/model/lingbot-va
export RUN_ROOT=/workspace/runs/lingbot-va
bash model_examples/LingBot-VA/run_8card.sh
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
