# 快速开始

本指南从 TurboPhysAI 已完成安装并通过检查开始，说明如何对模型启用优化。产品镜像和源码安装方法见[安装指南](installation.md)。

![TurboPhysAI 快速开始流程](../../assets/turbophysai-workflow.svg)

## 1. 确认前置条件

开始前应满足：

- 当前 Python 环境能够导入 `turbo_physai` 和 `turbo_physai.ops`；
- 目标 Python 训练入口及其运行依赖已经准备完成。

安装和环境检查方法见[安装指南](installation.md)。

## 2. 启动首次训练

`turbo-physai run` 是训练用户的默认入口。通过 `--model` 选择内置模型，并将原 Python 训练命令写在 TurboPhysAI 参数之后：

```bash
turbo-physai run \
  --model <model-name> \
  --log-report \
  python path/to/train.py
```

`<model-name>` 应替换为[模型支持清单](../models/support_list.md)中的模型名称。指定内置模型时，Runner 加载对应的 OptimizationConfig 和 RuntimeConfig。不指定模型和配置时，Runner 仅加载随包交付的公共 OptimizationConfig。训练命令及其参数由 Runner 原样执行；示例中的 `python path/to/train.py` 应替换为实际训练入口。

使用外部优化包或自定义配置时，可以将 `--model` 替换为 `--optimization-config` 和可选的 `--runtime-config`。完整命令参数和配置选择顺序见[CLI 参考](../reference/cli.md)。

### 使用 Python API

能够控制训练入口时，可以在导入目标模型模块前调用 `apply()`：

```python
import turbo_physai

report = turbo_physai.apply(
    model="bevformer",
    log_report=True,
)
```

示例中的 `bevformer` 可以替换为[模型支持清单](../models/support_list.md)中的其他模型名称。仅使用公共优化时省略 `model`；使用显式 OptimizationConfig 时，将 `model` 替换为 `optimization_config_path="path/to/optimization.yaml"`。

`apply()` 每个训练进程只能调用一次，只负责应用 OptimizationConfig，不加载 RuntimeConfig，也不负责启动训练命令。需要 RuntimeConfig 时应使用 `turbo-physai run`，或由调用方在执行模型代码前准备运行环境。完整参数、返回值和异常行为见[Python API 参考](../reference/python_api.md)。

## 3. 通过日志确认优化状态

`--log-report` 用于输出本次优化应用结果，默认关闭。启用后，Rank 0 在训练日志中输出完整 OptimizationReport，报告以以下标记界定：

```text
TURBO_PHYSAI_OPTIMIZATION_REPORT_BEGIN run_id=<run-id>
...
TURBO_PHYSAI_OPTIMIZATION_REPORT_END run_id=<run-id>
```

确认：

- 预期启用的 Group 为 `applied`；
- `blocked`、`failed`、`rolled_back` 和 `not_started` 均为 0；
- `skipped` 仅包含配置未启用、命令行临时禁用或依赖项被禁用的 Group，具体原因见报告中的 `reason`。

其他 rank 输出一行状态摘要。不需要查看应用详情时，可以从启动命令中删除 `--log-report`。OptimizationReport 只说明优化应用状态，不代替模型正确性、精度和性能验证。报告内容和状态定义见[优化应用报告](../user_guide/report.md)。

## 4. 后续阅读

- RuntimeConfig 字段和覆盖关系：[RuntimeConfig 使用指南](../user_guide/runtime_config.md)
- OptimizationConfig 字段和选择规则：[OptimizationConfig 使用指南](../user_guide/optimization_config.md)
- 内置模型和优化接入基线：[模型支持清单](../models/support_list.md)
- 完整 CLI 参数：[CLI 参考](../reference/cli.md)
- Python 调用接口：[Python API 参考](../reference/python_api.md)
- 报告字段和故障含义：[优化应用报告](../user_guide/report.md)
