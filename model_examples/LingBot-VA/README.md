# LingBot-VA 应用说明

本文说明如何在 VLA 配套镜像中，基于 LingBot-VA 官方仓库接入 TurboPhysAI 的 HCU 优化并启动训练。

## 1. 模型简介

[LingBot-VA](https://github.com/Robbyant/lingbot-va) 联合建模视频世界状态和机器人动作。本接入面向官方 LIBERO 后训练配置。

## 2. 优化接入基线

TurboPhysAI 的 LingBot-VA 优化基于官方仓库 commit `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

### 3.1 创建容器并安装 TurboPhysAI

请先按照[安装指南](https://github.com/HYGON-AI/TurboPhysAI/blob/main/docs/zh/get_started/installation.md#使用配套镜像)，使用 VLA 配套镜像创建容器并安装 TurboPhysAI。下文命令均在容器内执行。

### 3.2 安装 LingBot-VA 训练依赖

镜像已提供 HCU 版 PyTorch、Triton 和 FlashAttention。将本目录的 [requirements.txt](requirements.txt) 和 [run_8card.sh](run_8card.sh) 保存到宿主机工作目录的 `lingbot-va-tools/`，容器内对应 `/workspace/lingbot-va-tools/`。

```bash
cd /workspace/lingbot-va-tools
python3 -m pip install --no-deps -r requirements.txt
python3 -m pip check
```

使用 `--no-deps`，避免依赖解析替换镜像中的 HCU 基础包。

## 4. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/Robbyant/lingbot-va.git model/lingbot-va
cd model/lingbot-va
git checkout --detach 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb
git status --short
```

## 5. 准备 LIBERO 数据

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

## 6. 准备预训练权重

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

## 7. 通过 turbo-physai run 启动训练

`--model lingbot_va` 自动加载随包交付的九组 OptimizationConfig 和 NUMA RuntimeConfig。启动时使用 `turbo-physai run`，以便 Group 配置应用到训练进程及编译 worker。`linear_addmm` 保持启用，但 `action_embedder` 保留原始 `nn.Linear` 路径以避免 BF16 舍入顺序差异；其它白名单线性层仍使用融合实现。

### 7.1 单机八卡训练

```bash
cd /workspace/lingbot-va-tools
export LINGBOT_VA_ROOT=/workspace/model/lingbot-va
export RUN_ROOT=/workspace/runs/lingbot-va
bash run_8card.sh
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
