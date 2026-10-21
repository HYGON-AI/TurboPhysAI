# DiT4DiT 应用说明

本文说明如何在产品镜像中，基于官方 DiT4DiT 仓库应用 TurboPhysAI 的 HCU 优化。镜像获取和容器创建方法参见[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)。

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

产品基础镜像不包含 DiT4DiT 源码、模型权重或数据集。对外交付时应使用经过验证的完整镜像地址和不可变 digest，不使用 `latest`。

## 3. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/Mondo-Robotics/DiT4DiT.git model/DiT4DiT
cd model/DiT4DiT
git checkout --detach d8854a11713f846b8b3f768fc3bf2c6cba2bc17f
git status --short
```

`git status --short` 应无输出。下文命令均在容器内的 `/workspace/model/DiT4DiT` 执行。

## 4. 安装模型源码

```bash
cd /workspace/model/DiT4DiT
python3 -m pip install -e . --no-build-isolation --no-deps
```

产品镜像提供经过验证的 HCU/PyTorch 依赖。不要直接安装上游 `requirements.txt`，否则可能覆盖 HCU PyTorch 或引入 CUDA 依赖。

## 5. 准备模型权重与数据

按照 DiT4DiT 上游文档准备以下资源：

- Cosmos-Predict2.5-2B 模型及 VAE；
- Action 模型和 VLM 所需权重；
- RoboCasa 使用的 `PhysicalAI-Robotics-GR00T-X-Embodiment-Sim` 数据集。

下面的训练命令假设资源位于：

```text
/workspace/models/Cosmos-Predict2.5-2B-diffusers
/workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
/workspace/outputs
```

TurboPhysAI 不下载或重新分发模型权重与数据集。用户必须遵守对应资源的许可证和使用条款。

## 6. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改 DiT4DiT 模型源码即可应用优化。

### 6.1 单机双卡训练

```bash
cd /workspace/model/DiT4DiT

export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false

turbo-physai run \
  --model dit4dit \
  --log-report \
  accelerate launch \
    --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 2 \
    -m turbo_physai.optimizations.models.dit4dit.entry \
    --config_yaml DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml \
    --run_root_dir /workspace/outputs \
    --run_id dit4dit_hcu_2card \
    --framework.cosmos25.base_model \
      /workspace/models/Cosmos-Predict2.5-2B-diffusers \
    --datasets.vla_data.data_root_dir \
      /workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
    --datasets.vla_data.per_device_batch_size 2 \
    --trainer.gradient_accumulation_steps 4
```

该配置使用两个训练进程、每卡 batch size 2、梯度累积 4，对应 global batch size 16。RuntimeConfig 会依据 HCU 可见性为各 rank 自动选择本地 NUMA node。

产品化交付时，建议使用配套 `DiT4DiT-das` 仓库提供的镜像构建、容器启动和双卡训练脚本，避免遗漏设备、共享内存、模型路径和环境校验参数。

## 7. 优化范围与验收

默认 Recipe 原子应用 10 个 DiT4DiT Optimization Group，覆盖数据加载、Cosmos Attention/Cache/VAE/视频整理、Cosmos 分段编译、Action 编译、Fused AdamW 和训练指标同步。

Action Compile 与 `fallback_random=True` 位于同一个不可拆分 Group，用于保持 Dropout 等随机算子的 eager RNG 路径。Action 和 Cosmos 编译段固定使用当前 PyTorch 2.10 DAS 所需的 `alignment_asserts=False`；该选项只关闭 Inductor 生成代码的运行时对齐断言，不关闭 Flash-Attention，也不启用 TF32 或其他降精度选项。

正式验收至少应满足：

- OptimizationReport 中预期的 10 个 Group 全部为 `applied`；
- 双 rank 完成规定的 optimizer steps，Loss 无 NaN/Inf；
- Loss 与性能基线使用相同的 PyTorch、DTK、模型 commit、数据顺序、seed、batch size 和梯度累积；
- CPU/NUMA、MIOpen user DB 和 BLAS 产物与目标节点及镜像版本匹配；
- 记录 `action_dit_loss`、`future_video_loss`、`total_loss`、稳态 step time 和吞吐。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
