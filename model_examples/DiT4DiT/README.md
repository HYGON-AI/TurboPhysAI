# DiT4DiT 应用说明

## 1. 模型简介

[DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) 是面向通用机器人控制的 Vision-Action Model（VAM）框架，将视频生成 Transformer 与基于 Flow Matching 的动作预测联合建模，支持桌面操作和人形机器人全身控制。模型和训练方法的详细说明参见 [DiT4DiT 项目主页](https://dit4dit.github.io/)与[论文](https://arxiv.org/abs/2603.10448)。

## 2. 优化接入基线

TurboPhysAI 的 DiT4DiT 优化基于官方仓库 commit `d8854a11713f846b8b3f768fc3bf2c6cba2bc17f` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 下载并启动基础镜像

```bash
export BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10-20260929
docker pull "$BASE_IMAGE"

export WORKSPACE=/path/to/dit4dit-workspace
mkdir -p "$WORKSPACE"

docker run -dit \
  --name dit4dit \
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

docker exec -it dit4dit bash
```

## 4. 安装 TurboPhysAI 和模型依赖

```bash
cd /workspace
git clone https://github.com/HYGON-AI/TurboPhysAI.git
cd TurboPhysAI

export TURBO_PHYSAI_ROOT=/workspace/TurboPhysAI

PATH=/usr/bin:$PATH MAX_JOBS=8 python3 setup.py bdist_wheel
python3 -m pip install --no-deps dist/turbo_physai-*.whl
python3 -m pip install --no-deps \
  -r "$TURBO_PHYSAI_ROOT/model_examples/DiT4DiT/requirements.txt"
```

`--no-deps` 用于保护基础镜像中的 HCU PyTorch、DeepSpeed、Triton 等适配组件。不要直接安装 DiT4DiT 上游 `requirements.txt`，否则可能覆盖这些组件或引入 CUDA 依赖。

## 5. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/Mondo-Robotics/DiT4DiT.git model/DiT4DiT
cd model/DiT4DiT
git checkout --detach d8854a11713f846b8b3f768fc3bf2c6cba2bc17f
git status --short
```

```bash
python3 -m pip check
python3 -c "import torch, accelerate, diffusers, transformers; print(torch.__version__)"
turbo-physai --help
```

除基础镜像已有的 PyGObject 可选依赖，以及 Decord、pipablepytorch3d 的 wheel 平台元数据提示外，`pip check` 不应出现其他问题。已验证的关键版本如下：

| 依赖 | HCU 交付版本 | DiT4DiT 上游版本 |
| --- | --- | --- |
| PyTorch | 2.10 DAS/HIP | 2.6.0 |
| Accelerate | 1.13.0 | 1.12.0 |
| Diffusers | 0.38.0 | Git commit `3996788b602eaae4da41a1d45726b62e662b73cf` |
| Transformers | 4.57.1 | 4.57.6 |

Accelerate 1.13.0 和 Diffusers 0.38.0 是经过双卡 HCU 训练验证的组合；TurboPhysAI 的 Cosmos 优化直接适配 Diffusers 0.38.0，不能单独替换为上游 commit。该 RoboCasa 训练路径不安装未使用的 PEFT。

## 6. 准备模型权重与数据

- [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)，使用 `diffusers/base/post-trained` revision。该目录已包含 text encoder、VAE 和 transformer，无需单独下载 VAE/VLM 权重。
- [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) Action Tokenizer。
- [PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) 数据集，实际 data mix 为 `fourier_gr1_unified_1000`。
- 可选：[DiT4DiT-RoboCasa-GR1 预训练检查点](https://huggingface.co/TeliMa/dit4dit_robocasa_gr1)。从头训练不需要该检查点。

```bash
mkdir -p /workspace/models
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir /workspace/models/Cosmos-Predict2.5-2B-diffusers
```

```bash
cd /workspace/model/DiT4DiT
python3 examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py

mkdir -p /workspace/dataset
ln -s "$PWD/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim" \
  /workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 默认加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)。

### 7.1 生成并复用 MIOpen 缓存

将 MIOpen user DB 和 kernel cache 放在已挂载的 `/workspace` 中：

```bash
export DIT4DIT_CACHE_ROOT=/workspace/cache/dit4dit
mkdir -p \
  "$DIT4DIT_CACHE_ROOT/miopen/user_db" \
  "$DIT4DIT_CACHE_ROOT/miopen/kernel_cache"

export MIOPEN_USER_DB_PATH="$DIT4DIT_CACHE_ROOT/miopen/user_db"
export MIOPEN_CUSTOM_CACHE_DIR="$DIT4DIT_CACHE_ROOT/miopen/kernel_cache"

unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE

# 首次针对真实 Conv3D shape 生成 User PerfDb 时取消下面两行注释：
# export MIOPEN_FIND_MODE=NORMAL
# export MIOPEN_FIND_ENFORCE=SEARCH
# 需要重新搜索并更新数据库时，将 SEARCH 改为 SEARCH_DB_UPDATE。
```

首次搜索会明显变慢。后续训练只保留两个缓存路径。

### 7.2 单机双卡训练

```bash
cd /workspace/model/DiT4DiT

export WANDB_MODE=disabled

turbo-physai run \
  --model dit4dit \
  --log-report \
  accelerate launch \
    --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 2 \
    -m turbo_physai.optimizations.models.dit4dit.entry \
    --config_yaml DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml \
    --framework.name DiT4DiT \
    --framework.cosmos25.base_model \
      /workspace/models/Cosmos-Predict2.5-2B-diffusers \
    --framework.action_model.action_model_type DiT-B \
    --framework.cosmos25.extract_layer 17 \
    --framework.cosmos25.flow_matching.time_distribution uniform \
    --framework.cosmos25.flow_matching.high_sigma_ratio null \
    --framework.cosmos25.flow_matching.high_sigma_min null \
    --framework.cosmos25.conditional_frame_timestep 0.0001 \
    --datasets.vla_data.data_root_dir \
      /workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --datasets.vla_data.data_mix fourier_gr1_unified_1000 \
    --datasets.vla_data.per_device_batch_size 2 \
    --datasets.vla_data.action_video_freq_ratio 2 \
    --trainer.freeze_modules \
      backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae \
    --trainer.max_train_steps 200000 \
    --trainer.save_interval 50000 \
    --trainer.logging_frequency 10 \
    --trainer.eval_interval 100 \
    --trainer.learning_rate.backbone_interface 1e-5 \
    --trainer.learning_rate.action_model 1e-4 \
    --trainer.num_warmup_steps 5000 \
    --trainer.gradient_accumulation_steps 4 \
    --run_root_dir /workspace/outputs \
    --run_id dit4dit_hcu_2card
```

双卡配置为 2 进程 × 每卡 batch size 2 × 梯度累积 4，global batch size 为 16；学习率沿用上游 recipe。训练入口会同步 Accelerate 与 DeepSpeed 的梯度累积配置，RuntimeConfig 自动完成 rank/NUMA 映射。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
