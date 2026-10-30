# DiT4DiT 应用说明

本文说明如何在 HCU 基础镜像中安装 DiT4DiT 依赖和 TurboPhysAI，并基于官方 DiT4DiT 仓库应用优化。

## 1. 模型简介

[DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) 是面向通用机器人控制的 Vision-Action Model（VAM）框架，将视频生成 Transformer 与基于 Flow Matching 的动作预测联合建模，支持桌面操作和人形机器人全身控制。模型和训练方法的详细说明参见 [DiT4DiT 项目主页](https://dit4dit.github.io/)与[论文](https://arxiv.org/abs/2603.10448)。

## 2. 优化接入基线

TurboPhysAI 的 DiT4DiT 优化基于官方仓库 commit `d8854a11713f846b8b3f768fc3bf2c6cba2bc17f` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

已验证的产品环境为：

- Python 3.10；
- DTK 26.04；
- PyTorch 2.10；
- Diffusers 0.38.0；
- Accelerate 1.13.0。



## 3. 启动基础镜像

基础镜像只提供 DTK、HCU PyTorch、DeepSpeed 等平台依赖。已验证镜像如下：

```bash
export BASE_IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10-20260929
export CONTAINER_NAME=dit4dit-hcu
export WORKSPACE=/path/to/dit4dit-workspace

mkdir -p "$WORKSPACE"
docker pull "$BASE_IMAGE"
docker run -dit \
  --name "$CONTAINER_NAME" \
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
  "$BASE_IMAGE" sleep infinity

docker exec -it "$CONTAINER_NAME" bash
```

## 4. 安装 TurboPhysAI 和模型依赖

将正式 wheel 放入 `/workspace/packages`，并准备与 wheel 相同版本的 TurboPhysAI 仓库：

```bash
cd /workspace
git clone https://github.com/HYGON-AI/TurboPhysAI.git
cd TurboPhysAI
git checkout --detach RELEASE_TAG_OR_COMMIT

export TURBO_PHYSAI_ROOT=/workspace/TurboPhysAI
export TURBO_PHYSAI_WHEEL=/workspace/packages/turbo_physai-VERSION-py3-none-any.whl

python3 -m pip install --no-deps "$TURBO_PHYSAI_WHEEL"
python3 -m pip install --no-deps \
  -r "$TURBO_PHYSAI_ROOT/model_examples/DiT4DiT/requirements.txt"
```

`--no-deps` 用于保护基础镜像中的 HCU PyTorch、DeepSpeed、Triton 等适配组件。不要直接安装 DiT4DiT 上游 `requirements.txt`，否则可能覆盖这些组件或引入 CUDA 依赖。

## 5. 准备并安装模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/Mondo-Robotics/DiT4DiT.git model/DiT4DiT
cd model/DiT4DiT
git checkout --detach d8854a11713f846b8b3f768fc3bf2c6cba2bc17f
git status --short
python3 -m pip install -e . --no-build-isolation --no-deps
```

`git status --short` 应无输出。模型源码同样使用 `--no-deps` 安装。随后检查组合环境：

```bash
python3 -m pip check
python3 -c "import torch, accelerate, diffusers, transformers; print(torch.__version__)"
turbo-physai --help
```

除基础镜像已有的 PyGObject 可选依赖，以及 Decord、pipablepytorch3d 的 wheel 平台元数据提示外，`pip check` 不应出现其他问题。已验证的关键版本如下：

| 依赖 | HCU 交付版本 | DiT4DiT 上游版本 |
| --- | --- | --- |
| PyTorch | 2.10 DAS/HIP | 2.6.0 |
| DeepSpeed | 0.18.4 DAS | 0.18.4 |
| Accelerate | 1.13.0 | 1.12.0 |
| Diffusers | 0.38.0 | Git commit `3996788b602eaae4da41a1d45726b62e662b73cf` |
| OmegaConf | 2.3.0 | 2.3.0 |
| Decord | 0.6.0 | 0.6.0 |
| Transformers | 4.57.1 | 4.57.6 |

Accelerate 1.13.0 和 Diffusers 0.38.0 是经过双卡 HCU 训练验证的组合；TurboPhysAI 的 Cosmos 优化直接适配 Diffusers 0.38.0，不能单独替换为上游 commit。该 RoboCasa 训练路径不安装未使用的 PEFT。

## 6. 准备模型权重与数据

训练使用以下资源：

- [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)，使用 `diffusers/base/post-trained` revision。该目录已包含 text encoder、VAE 和 transformer，无需单独下载 VAE/VLM 权重。
- [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast) Action Tokenizer。
- [PhysicalAI-Robotics-GR00T-X-Embodiment-Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) 数据集，实际 data mix 为 `fourier_gr1_unified_1000`。
- 可选：[DiT4DiT-RoboCasa-GR1 预训练检查点](https://huggingface.co/TeliMa/dit4dit_robocasa_gr1)。从头训练不需要该检查点。

下载 Cosmos 权重：

```bash
mkdir -p /workspace/models
huggingface-cli download nvidia/Cosmos-Predict2.5-2B \
  --revision diffusers/base/post-trained \
  --local-dir /workspace/models/Cosmos-Predict2.5-2B-diffusers
```

按上游脚本下载 RoboCasa 使用的数据子集，并将下载目录映射到下文训练路径：

```bash
cd /workspace/model/DiT4DiT
python3 examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py

mkdir -p /workspace/dataset
ln -s "$PWD/playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim" \
  /workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
```

如果目标链接已经存在，请直接复用现有目录，不要重复创建。

下面的训练命令假设资源位于：

```text
/workspace/models/Cosmos-Predict2.5-2B-diffusers
/workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
/workspace/outputs
```


## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改 DiT4DiT 模型源码即可应用优化。使用自定义配置时，通过 `--optimization-config /path/to/optimization.yaml` 和 `--runtime-config /path/to/runtime.yaml` 显式指定文件。

### 7.1 生成并复用 MIOpen 缓存

将 MIOpen user DB 和 kernel cache 放在已挂载的 `/workspace` 中：

```bash
export DIT4DIT_CACHE_ROOT=/workspace/cache/dit4dit
mkdir -p \
  "$DIT4DIT_CACHE_ROOT/miopen/user_db" \
  "$DIT4DIT_CACHE_ROOT/miopen/kernel_cache"

export MIOPEN_USER_DB_PATH="$DIT4DIT_CACHE_ROOT/miopen/user_db"
export MIOPEN_CUSTOM_CACHE_DIR="$DIT4DIT_CACHE_ROOT/miopen/kernel_cache"
```

首次针对真实 Conv3D shape 生成 User PerfDb 时，在执行 7.2 的训练命令前设置：

```bash
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_FIND_ENFORCE=SEARCH
```

该次运行会搜索候选实现，因此明显更慢。数据库生成后，正常训练仅保留两个缓存路径并取消搜索变量：

```bash
unset MIOPEN_FIND_MODE
unset MIOPEN_FIND_ENFORCE
```

不要使用 `SEARCH_DB_UPDATE`，也不要设置 TF32 或模型专用 BLAS/Tensile 环境变量。正式性能测试应在数据库生成后重新启动，并先完成 warmup。

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

除硬件规模相关参数外，上述模型与训练 recipe 与上游 RoboCasa 训练脚本保持一致。为适配双卡 HCU 环境，示例将训练进程数由 16 调整为 2、每卡 batch size 由 4 调整为 2，并将梯度累积由 1 调整为 4，对应 global batch size 16。打包入口 `turbo_physai.optimizations.models.dit4dit.entry` 用于将梯度累积配置同步到 Accelerate 和 DeepSpeed；text encoder 和 VAE 仍按上游 recipe 保持冻结。RuntimeConfig 会依据 HCU 可见性为各 rank 自动选择本地 NUMA node。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
