# FastWAM 应用说明

本文说明如何基于官方 FastWAM 仓库应用 TurboPhysAI 优化。请先按照[安装指南](../../docs/zh/get_started/installation.md)安装 TurboPhysAI，并准备与设备匹配的 PyTorch、Accelerate 和 DeepSpeed 环境。

## 1. 模型简介

[FastWAM](https://github.com/yuantianyuan01/FastWAM) 基于 Wan2.2 视频模型，通过视频与动作专家的混合 Transformer（MoT）联合建模未来视频和机器人动作。TurboPhysAI 提供 DiT、MoT、VAE 编码及推理路径的优化，支持 FastWAM、Joint 和 IDM 变体。

## 2. 优化接入基线

TurboPhysAI 的 FastWAM 优化基于官方仓库 commit `7faa71108368fbb3b6885649f112af607427a2d4` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备模型源码

```bash
mkdir -p /workspace/model
cd /workspace/model
git clone https://github.com/yuantianyuan01/FastWAM.git
cd /workspace/model/FastWAM
git checkout --detach 7faa71108368fbb3b6885649f112af607427a2d4
python -m pip install --no-deps --no-build-isolation -e .
mkdir -p checkpoints data runs
```

按原模型仓库的安装说明准备其余依赖。下文命令均在上述环境和模型仓库根目录中执行。

## 4. 准备 LIBERO 数据

从 [LIBERO-fastwam 数据集](https://huggingface.co/datasets/yuanty/LIBERO-fastwam) 下载并解压原模型使用的 LeRobot 数据，目录结构应为：

```text
data/libero_mujoco3.3.2/
├── libero_10_no_noops_lerobot/
├── libero_goal_no_noops_lerobot/
├── libero_object_no_noops_lerobot/
└── libero_spatial_no_noops_lerobot/
```

使用其他存储位置时，修改原仓库 `configs/data/libero_2cam.yaml` 中的 `train.dataset_dirs`。

## 5. 准备预训练权重和文本缓存

按原模型仓库说明准备 Wan2.2-TI2V-5B 及文本编码器权重，默认存放在 `checkpoints/`。生成 ActionDiT 初始权重：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

训练前预计算任务使用的文本 embedding：

```bash
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

缓存默认写入 `data/text_embeds_cache/libero`，训练时通过原模型配置读取。

## 6. 通过 turbo-physai run 启动训练

`turbo-physai run --model fastwam` 自动加载内置 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md)，在各训练 rank 中应用优化，无需修改原模型训练入口。需要自定义运行环境时，通过 `--runtime-config` 指定 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)。

```bash
turbo-physai run \
  --model fastwam \
  --log-report \
  python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
    --num_processes 8 --num_machines 1 \
    scripts/train.py \
    task=libero_uncond_2cam224_1e-4 \
    batch_size=12 \
    model.mot_checkpoint_mixed_attn=true \
    model.action_scheduler.train_shift=5.0 \
    model.action_scheduler.infer_shift=5.0
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
