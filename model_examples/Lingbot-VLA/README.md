# LingBot-VLA 应用说明

## 1. 模型简介

[LingBot-VLA](https://github.com/Robbyant/lingbot-vla) 是 4B 规模的开源视觉-语言-动作（VLA）模型，以 Qwen2.5-VL-3B 作为视觉语言主干，配合 action expert 统一支持多种双臂机器人动作空间。训练以动作空间上的流匹配（Flow Matching）为主任务，使用深度预测作为代理任务，MoGe 提供相对深度、LingBot-Depth 提供几何特征。训练入口脚本为 `tasks/vla/train_lingbotvla.py`。

## 2. 优化接入基线

TurboPhysAI 的 LingBot-VLA 优化基于官方仓库 commit `4eb34b7693a0565c67433f8fac9c59a2e67eb60b` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

### 3.1 创建容器并安装 TurboPhysAI

按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 VLA 配套镜像创建容器并安装 TurboPhysAI。下文命令均在容器内执行。

### 3.2 安装 LingBot-VLA 训练依赖

VLA 配套镜像已提供 DTK 版 torch 等基础包，LingBot-VLA 训练还需要安装额外依赖，清单见 [requirements.txt](requirements.txt)。

```bash
# 安装训练依赖
pip install -r /path/to/TurboPhysAI/model_examples/Lingbot-VLA/requirements.txt

# lerobot 只从源码安装，避免 pip 解析出与镜像不兼容的依赖
pip install https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz --no-deps

# 安装几何工具库，官方锁定该 commit
pip install "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" --no-deps
```

## 4. 准备模型源码

```bash
cd /workspace
git clone https://github.com/Robbyant/lingbot-vla.git
cd /workspace/lingbot-vla
git checkout --detach 4eb34b7693a0565c67433f8fac9c59a2e67eb60b
git submodule update --init --recursive --remote

pip install -e /workspace/lingbot-vla --no-deps
pip install -e /workspace/lingbot-vla/lingbotvla/models/vla/vision_models/lingbot-depth/ --no-deps
pip install -e /workspace/lingbot-vla/lingbotvla/models/vla/vision_models/MoGe/ --no-deps
```

## 5. 准备 RoboTwin 2.0 数据

LingBot-VLA 使用 RoboTwin 2.0 数据集（LeRobot 格式）。数据集的下载与预处理步骤如下：

1. 参考官方文档 [RoboTwin2.0 Preparation](https://github.com/Robbyant/lingbot-vla/blob/main/experiment/robotwin/README.md) 生成 LeRobot V2.1 格式的数据集。

   注：官方 `generate.sh` 默认把数据集写到 lerobot 缓存目录 `${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/lerobot/<repo_id>`（`XDG_CACHE_HOME` 未设置时即 `~/.cache`）。

2. 将生成好的数据集移动到工作目录，后续转换、训练都使用这里的路径：

  ```bash
  mkdir -p /workspace/dataset/RoboTwin2.0/lerobot
  for repo_id in robotwin_adjust_bottle robotwin_place_object_stand; do
    mv ${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/lerobot/${repo_id} /workspace/dataset/RoboTwin2.0/lerobot/
  done
  ```

3. 多任务训练时，按官方 README 的 mixed 流程，先用仓库自带的 `scripts/merge_lerobot_v21.py` 把各任务的 V2.1 数据合并成一个数据集，再用 lerobot 自带的脚本转换为 V3.0 格式：

  ```bash
  # 合并多个任务的 V2.1 数据集
  python scripts/merge_lerobot_v21.py \
    --sources /workspace/dataset/RoboTwin2.0/lerobot/robotwin_adjust_bottle,/workspace/dataset/RoboTwin2.0/lerobot/robotwin_place_object_stand \
    --output /workspace/dataset/RoboTwin2.0/mixed_robotwin_5tasks

  # --root 传数据集父目录（脚本自动拼接 <repo_id>），--repo-id 传数据集目录名
  python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id=mixed_robotwin_5tasks \
    --root=/workspace/dataset/RoboTwin2.0 \
    --push-to-hub=false
  ```

  转换是原地进行的：转完后 `mixed_robotwin_5tasks` 即为 v3.0 数据，转换前的 v2.1 数据被改名为 `mixed_robotwin_5tasks_old`（确认转换完成后可以删除）。

  只训练单个任务时，跳过合并步骤，直接对单个数据集执行上面的 V3.0 转换。

4. `--data.train_path` 指向转换后的 V3.0 数据集目录，例如 `/workspace/dataset/RoboTwin2.0/mixed_robotwin_5tasks`。

## 6. 准备模型权重

后训练需使用官方脚本下载以下权重。脚本会在 `--local_dir` 下自动创建以仓库名命名的子目录：

```bash
# LingBot-VLA-4B 深度蒸馏版预训练权重（含 depth teacher）
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-vla-4b-depth --local_dir /workspace/models

# Qwen2.5-VL-3B-Instruct tokenizer
python3 scripts/download_hf_model.py --repo_id Qwen/Qwen2.5-VL-3B-Instruct --local_dir /workspace/models

# MoGe-2 深度模型
python3 scripts/download_hf_model.py --repo_id Ruicheng/moge-2-vitb-normal --local_dir /workspace/models

# LingBot-Depth 几何模型
python3 scripts/download_hf_model.py --repo_id robbyant/lingbot-depth-pretrain-vitl-14 --local_dir /workspace/models
```

各权重在训练配置 `./configs/vla/robotwin_load20000h_depth.yaml` 中的名称如下，请**直接修改配置文件**填入本机路径：

| 配置名称 | 本机路径 |
| :--- | :--- |
| model.model_path | /workspace/models/lingbot-vla-4b-depth |
| model.tokenizer_path | /workspace/models/Qwen2.5-VL-3B-Instruct |
| model.moge_path | /workspace/models/moge-2-vitb-normal/model.pt |
| model.morgbd_path | /workspace/models/lingbot-depth-pretrain-vitl-14/model_mdm_pre.pt |

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 LingBot-VLA 优化。

### 7.1 快速验证（单机八卡，100 step）

```bash
turbo-physai run \
  --model lingbot_vla \
  --log-report \
  -- \
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h_depth.yaml \
  --data.train_path /workspace/dataset/RoboTwin2.0/mixed_robotwin_5tasks \
  --data.norm_stats_file assets/norm_stats/robotwin_50.json \
  --train.max_steps 100 \
  --train.output_dir /workspace/runs/robotwin_100step
```

### 7.2 完整单机八卡训练

```bash
turbo-physai run \
  --model lingbot_vla \
  --log-report \
  -- \
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  tasks/vla/train_lingbotvla.py ./configs/vla/robotwin_load20000h_depth.yaml \
  --data.train_path /workspace/dataset/RoboTwin2.0/mixed_robotwin_5tasks \
  --data.norm_stats_file assets/norm_stats/robotwin_50.json \
  --train.output_dir /workspace/runs/robotwin
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
