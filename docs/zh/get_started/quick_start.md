# 快速开始

本指南从已安装 TurboPhysAI 开始，说明如何对模型启用优化。

![TurboPhysAI 快速开始流程](../../assets/turbophysai-workflow.svg)

## 1. 确认前置条件

开始前应满足：

- 模型仓库处于 TurboPhysAI 支持的 commit；
- 未使用 TurboPhysAI 时，原训练命令能够正常启动；
- 数据集已经按照模型官方说明完成预处理；
- 配置文件和预训练权重已经准备完成；
- TurboPhysAI wheel 与当前 HCU/PyTorch 环境匹配。

内置模型和优化接入基线见[模型支持清单](../models/support_list.md)。

## 2. 选择模型

每个内置模型均随包提供优化配置（OptimizationConfig）和运行配置（RuntimeConfig）。通过 `--model <模型名>` 选择模型后，Runner 自动加载对应配置，无需查找包内文件路径。`--model` 的参数就是模型名，不区分大小写，例如 `--model bevformer`。

支持的模型名、优化接入基线和使用说明统一见[模型支持清单](../models/support_list.md)。

## 3. 启动训练

`turbo-physai run` 是训练用户的默认入口。该命令根据 `--model` 或显式配置路径选择 OptimizationConfig 和 RuntimeConfig，然后执行后续的原训练命令。

```bash
turbo-physai run \
  --model bevformer \
  --log-report \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

Runner 自动加载 `bevformer` 随包交付的 OptimizationConfig 和 RuntimeConfig。不指定模型时，Runner 使用内置公共 OptimizationConfig，不加载模型专用优化或 RuntimeConfig：

```bash
turbo-physai run \
  python tools/train.py path/to/config.py
```

外部优化包或自定义交付可以显式传入 `--optimization-config` 和 `--runtime-config`。显式路径优先于 `--model` 自动选择的配置。

`--nproc-per-node` 是当前节点的训练进程数；设为 `1` 可进行单卡冒烟，设为 `8` 可启动八卡训练。NUMA 默认开启；不需要绑定时，在 `turbo-physai run` 参数中增加 `--disable-numa`。

### 高级用法：Python API

能够控制训练入口时，可以在导入模型相关模块前调用 `apply()`：

```python
import turbo_physai

report = turbo_physai.apply(
    model="bevformer",
    log_report=True,
)
```

`log_report` 默认值为 `False`。设为 `True` 时，Rank 0 将完整 OptimizationReport 输出到标准日志；无论是否输出日志，`apply()` 都会返回报告对象。每个训练进程只能调用一次 `apply()`。

## 4. 查看执行结果

`--log-report` 用于输出本次优化应用结果，默认关闭。启用后，Rank 0 在训练日志中输出完整 OptimizationReport，报告以以下标记界定：

```text
TURBO_PHYSAI_OPTIMIZATION_REPORT_BEGIN run_id=<run-id>
...
TURBO_PHYSAI_OPTIMIZATION_REPORT_END run_id=<run-id>
```

确认：

- 预期启用的 Group 为 `applied`；
- `blocked`、`failed`、`rolled_back` 和 `not_started` 均为 0；
- `skipped` 仅包含配置未启用、命令行临时禁用或依赖项被禁用的 Group，具体原因见报告中的 `reason`；
- 模型、数据集和预训练权重正常加载；
- 首个训练迭代能够完成，loss 没有 NaN/Inf；
- 预热后迭代时间稳定；
- 按模型应用说明完成精度和性能验证。

其他 rank 输出一行状态摘要。不需要查看应用详情时，可以从启动命令中删除 `--log-report`。报告内容和状态定义见[优化应用报告](../user_guide/report.md)。

## 5. 后续阅读

- RuntimeConfig 字段和覆盖关系：[RuntimeConfig 使用指南](../user_guide/runtime_config.md)
- OptimizationConfig 字段和选择规则：[OptimizationConfig 使用指南](../user_guide/optimization_config.md)
- 完整 CLI 参数：[CLI 参考](../reference/cli.md)
- 报告字段和故障含义：[优化应用报告](../user_guide/report.md)
