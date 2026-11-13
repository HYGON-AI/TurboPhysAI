# LingBot-VLA-V2 应用说明

## 1. 模型简介

[LingBot-VLA-V2](https://github.com/Robbyant/lingbot-vla-v2) 是 6B 规模的开源视觉-语言-动作（VLA）模型，以 Qwen3-VL-4B 作为视觉语言主干，配合 MoE action expert，统一支持手臂、末端执行器、夹爪、灵巧手、腰、头与移动底盘等动作空间。训练以动作空间上的流匹配（Flow Matching）为主任务，使用未来帧预测与深度预测为代理任务，DINO-VIDEO teacher 提供语义时序先验、LingBot-Depth 提供几何线索。训练入口脚本为 `tasks/vla/train_lingbotvla.py`。

## 2. 优化接入基线

TurboPhysAI 的 LingBot-VLA-V2 优化基于官方仓库 commit `be969b8fd117fb70550c5d4bf4bc328211b5b1b6` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。


## 3. 构建训练环境

### 3.1 创建容器并安装 TurboPhysAI

按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 VLA 配套镜像创建容器并安装 TurboPhysAI。下文命令均在容器内执行。

### 3.2 安装 LingBot-VLA-V2 训练依赖

镜像已提供 DTK 版 torch 等基础包，LingBot-VLA-V2 训练还需要安装额外依赖，清单见 [requirements.txt](requirements.txt)。

```bash
# 安装训练依赖
pip install -r /path/to/TurboPhysAI/model_examples/Lingbot-VLA-V2/requirements.txt

# lerobot 只从源码安装，避免 pip 解析出与镜像不兼容的依赖
pip install lerobot==0.4.4 --no-deps

# 安装几何工具库，官方锁定该 commit
pip install "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" --no-deps
```

## 4. 准备模型源码

```bash
cd /workspace
git clone https://github.com/Robbyant/lingbot-vla-v2.git
cd /workspace/lingbot-vla-v2
git checkout --detach be969b8fd117fb70550c5d4bf4bc328211b5b1b6

pip install -e /workspace/lingbot-vla-v2 --no-deps
```

## 5. 准备 RoboTwin 2.0 数据

LingBot-VLA-V2 使用 RoboTwin 2.0 数据集（LeRobot 格式）。数据集的下载与预处理见官方文档 [RoboTwin2.0 Preparation](https://github.com/Robbyant/lingbot-vla-v2/blob/main/experiment/robotwin/README.md)。

数据集目录结构示例：

```text
/workspace/dataset/RoboTwin2.0/lerobot
├── robotwin_adjust_bottle
├── robotwin_place_object_stand
└── ...
```

仓库自带的 `assets/training_data/robotwin.txt` 中路径为占位符，使用前需替换为实际数据集路径，格式为 `<robot_config 名> <数据集绝对路径>`：

```text
robotwin /workspace/dataset/RoboTwin2.0/lerobot/robotwin_adjust_bottle
robotwin /workspace/dataset/RoboTwin2.0/lerobot/robotwin_place_object_stand
...
```

## 6. 准备模型权重

后训练需使用官方脚本下载以下权重。脚本会在 `--local_dir` 下自动创建以仓库名命名的子目录：

```bash
# LingBot-VLA-V2 6B 预训练权重（含 depth 与 dino_video teacher）
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-v2-6b --local_dir /workspace/models

# Qwen3-VL-4B-Instruct tokenizer
python3 scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-4B-Instruct --local_dir /workspace/models

# MoGe-2 深度模型
python3 scripts/download_hf_model.py --repo_id Ruicheng/moge-2-vitb-normal --local_dir /workspace/models
```

各权重在训练配置`./configs/vla/robotwin/robotwin.yaml`中的名称如下，请**直接修改配置文件**填入本机路径：

| 配置名称 | 本机路径 |
| :--- | :--- |
| model.model_path | /workspace/models/lingbot-vla-v2-6b |
| model.tokenizer_path | /workspace/models/Qwen3-VL-4B-Instruct |
| train.align_params.depth.moge_path | /workspace/models/moge-2-vitb-normal/model.pt |
| train.align_params.depth.morgbd_path | /workspace/models/lingbot-vla-v2-6b/depth/model.pt |
| train.align_params.video.ckpt_path | /workspace/models/lingbot-vla-v2-6b/dino_video/teacher_step_10000.pth |
| train.align_params.video.config_path | /workspace/models/lingbot-vla-v2-6b/dino_video/config.yaml |

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 LingBot-VLA-V2 优化。

### 7.1 快速验证（单机八卡，100 step）

```bash
# 官方脚本是32个GPU训练， global_bs 1024 = 32 * micro_bs 32，8卡训练时需要传入 --train.global_batch_size 256
# 并且要在BW1000上保持与官方一致的micro bs 32需要传入--train.enable_gradient_checkpointing True
turbo-physai run \
  --model lingbot_vla_v2 \
  --log-report \
  -- \
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing True \
  --train.max_steps 100 \
  --train.output_dir /workspace/runs/robotwin_100step
```

### 7.2 完整单机八卡训练

```bash
# 官方脚本是32个GPU训练， global_bs 1024 = 32 * micro_bs 32，8卡训练时需要传入 --train.global_batch_size 256
# 并且要在BW1000上保持与官方一致的micro bs 32需要传入--train.enable_gradient_checkpointing True
turbo-physai run \
  --model lingbot_vla_v2 \
  --log-report \
  -- \
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  tasks/vla/train_lingbotvla.py ./configs/vla/robotwin/robotwin.yaml \
  --train.global_batch_size 256 \
  --train.enable_gradient_checkpointing True \
  --train.output_dir /workspace/runs/robotwin
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
