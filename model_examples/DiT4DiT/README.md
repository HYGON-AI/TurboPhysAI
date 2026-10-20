# DiT4DiT 应用说明

本文说明如何在 HCU 产品环境中，对固定 DiT4DiT 基线应用 TurboPhysAI 优化。

## 1. 环境与代码基线

- 产品基础镜像：`dit-vla-wm-base:dtk26.04-torch2.10-py3.10`
- DiT4DiT commit：`d8854a11713f846b8b3f768fc3bf2c6cba2bc17f`
- Python：3.10
- DTK：26.04
- PyTorch：2.10

基础镜像不包含 DiT4DiT 源码、模型权重或数据集。对外交付时应使用已通过回归的完整 Harbor 镜像地址和 digest，不使用 `latest`。

## 2. 准备 DiT4DiT

```bash
cd /workspace
git clone https://github.com/DiT4DiT/DiT4DiT.git
cd DiT4DiT
git checkout --detach d8854a11713f846b8b3f768fc3bf2c6cba2bc17f
python -m pip install -e . --no-build-isolation --no-deps
```

产品镜像提供已验证的 HCU/PyTorch 依赖，不要直接安装上游 `requirements.txt`，避免覆盖 HCU PyTorch 或安装 CUDA 依赖。

## 3. 准备数据与权重

按 DiT4DiT 上游文档准备 RoboCasa 数据、Cosmos/VAE、Action 模型和 VLM 权重，并在 `DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml` 中设置实际路径。TurboPhysAI 不下载或重新分发数据和权重。

## 4. 双卡启动

```bash
cd /workspace/DiT4DiT

turbo-physai run \
  --model dit4dit \
  --log-report \
  accelerate launch \
    --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 2 \
    -m turbo_physai.optimizations.models.dit4dit.entry \
    --config_yaml DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml \
    --trainer.gradient_accumulation_steps 4
```

`--model dit4dit` 自动选择随 TurboPhysAI wheel 交付的 OptimizationConfig 和 RuntimeConfig。Action Compile、`fallback_random=True` 与当前 PyTorch 2.10 DAS 所需的 Action-scoped `alignment_asserts=False` 作为一个不可拆分的 Group 应用。该对齐开关只关闭 Action 编译生成代码的运行时对齐断言，不关闭 Flash-Attention。

## 5. 验收

正式结果必须同时满足：

- OptimizationReport 中预期的 10 个 Group 全部为 `applied`；
- 双 rank 无 NaN/Inf，完成规定的 optimizer steps；
- Loss 和性能基线来自同一 PyTorch 2.10、DTK、BLAS 和 MIOpen 环境；
- CPU/NUMA、MIOpen user DB 和 BLAS 产物与当前节点和镜像版本匹配。
