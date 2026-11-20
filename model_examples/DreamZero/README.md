# DreamZero 应用说明

本文说明如何在 VLA 配套镜像中，基于 DreamZero 官方仓库接入 TurboPhysAI 的 HCU 优化并启动训练。

## 1. 模型简介

[DreamZero](https://github.com/dreamzero0/dreamzero) 一个世界动作模型，能够联合预测动作和视频，在未见任务上实现强劲的zero-shot性能。本接入面向官方 DreamZero 在[DreamZero-DROID-Data](https://huggingface.co/datasets/GEAR-Dreams/DreamZero-DROID-Data)数据集上训练配置。

## 2. 优化接入基线

TurboPhysAI 的 DreamZero 优化基于官方仓库 commit `ab790c198fbce33503358efbbd4187ce9a89adf3` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

### 3.1 创建容器并安装 TurboPhysAI

请先按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 VLA 配套镜像创建容器并安装 TurboPhysAI。下文命令均在容器内执行。

### 3.2 安装 DreamZero 训练依赖

镜像已提供 HCU 版 PyTorch、Triton 和 FlashAttention，其余训练依赖见 [requirements.txt](requirements.txt)。将 `TURBO_PHYSAI_ROOT` 设置为容器内的 TurboPhysAI 仓库路径：

```bash
# 卸载多余会导致报错的包
pip uninstall -y transformer_engine transformer_engine_torch
export TURBO_PHYSAI_ROOT=/path/to/TurboPhysAI
python3 -m pip install -r "$TURBO_PHYSAI_ROOT/model_examples/DreamZero/requirements.txt"
python3 -m pip check
```

## 4. 准备模型源码

```bash
cd /workspace
git clone https://github.com/dreamzero0/dreamzero.git
cd dreamzero
git checkout --detach ab790c198fbce33503358efbbd4187ce9a89adf3
git status --short
```

## 5. 准备 DreamZero-DROID-Data 数据

下载官方预处理后的数据集（约 131GB）DreamZero-DROID-Data 数据集：

该数据集源自[DROID 1.0.1](https://droid-dataset.github.io/)数据集，并进行了以下修改：
- 从 RLDS/TFDS 格式转换为[LeRobot](https://github.com/huggingface/lerobot) 格式
- 使用[Physical Intelligence's idle frame detector](https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README_train.md#data-filtering) 移除空闲帧(`droid_sample_ranges_v1_0_1.json`)
没有语言注释的剧集会被过滤掉。
- 仅包含成功事件（奖励不为零的事件）
- 3 个摄像头视角：`exterior_image_1_left`, `exterior_image_2_left`, `wrist_image_left`

```bash
cd /workspace
mkdir -p /workspace/datasets
# You may need to set your HuggingFace token:
# export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
export HF_ENDPOINT=https://hf-mirror.com
hf download \
    GEAR-Dreams/DreamZero-DROID-Data \
    --repo-type dataset \
    --local-dir ./datasets/droid_lerobot
```

## 6. 准备预训练权重

下载Wan2.2 14B I2V模型权重(~28GB)，用于14B模型训练：

```bash
mkdir -p /workspace/checkpoints
modelscope download \
    --model Wan-AI/Wan2.1-I2V-14B-480P \
    --local_dir ./checkpoints/Wan2.1-I2V-14B-480P
```
使用较小的模型训练

下载Wan2.2 5B TI2V模型权重，WAN2.2-TI2V-5B 不包含 CLIP 图像编码器。使用 WAN2.2-I2V-14B-480P 里的那个，只需要 `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`

```bash
modelscope download \
    --model Wan-AI/Wan2.2-TI2V-5B \
    --local_dir ./checkpoints/Wan2.2-TI2V-5B
```

下载UMT5-XXL tokenizer

```bash
modelscope download \
    --model google/umt5-xxl \
    --local_dir ./checkpoints/umt5-xxl
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run --model DreamZero -- <训练命令>` 会从已安装的 TurboPhysAI 包的 `turbo_physai/optimizations/models/dreamzero/configs/` 目录自动加载 `optimization.yaml` 和 `runtime.yaml`。`--` 后传入原始训练命令，`--log-report` 等 TurboPhysAI 选项放在 `--` 前。

### 7.1 单机八卡训练

`run.sh` 基于官方 [`scripts/train/droid_training_full_finetune.sh`](../../../dreamzero/scripts/train/droid_training_full_finetune.sh) 的单机八卡启动方式，使用 DreamZero 5B 训练配置。脚本已包含 `--model DreamZero` 和 `--log-report`，可直接运行。

```bash
export OUTPUT_DIR=/workspace/runs/dreamzero_droid_wan22_full
bash "$TURBO_PHYSAI_ROOT/model_examples/DreamZero/run.sh"
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
