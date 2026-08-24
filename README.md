# TurboPhysAI

TurboPhysAI 是面向 Physical AI 领域模型（自动驾驶、具身智能、世界模型等）的训练性能优化组件，无需修改模型源码，即可通过算子、模型计算图、数据链路和训练过程优化，充分发挥 HCU 高性能计算能力。

关于组件的适用场景、能力边界和整体工作方式，参见 [TurboPhysAI 概述](docs/zh/overview.md)。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 公共性能优化 | 提供与具体模型上下文无关、入口稳定的算子和框架优化，供不同模型复用 |
| 模型专用优化 | 针对具体模型的算子、`forward`、数据链路、编译热点和训练过程进行组合优化 |

## 应用优化

推荐使用 `turbo-physai run` 启动原有训练命令。组件根据模型或配置参数选择优化能力，并在训练启动时完成应用。

### 使用已支持模型的专用优化

通过 `--model` 指定已支持的模型，组件会自动加载随包交付的模型专用优化和训练环境配置。以 BEVFormer 为例：

```bash
turbo-physai run \
  --model bevformer \
  -- \
  torchrun --nproc-per-node=8 tools/train.py \
    ./projects/configs/bevformer/bevformer_base.py \
    --launcher pytorch
```

### 使用默认公共优化

对于尚未提供模型专用优化的模型，不指定 `--model` 即可使用默认公共优化：

```bash
turbo-physai run -- python tools/train.py path/to/config.py
```

默认配置仅应用与当前软件环境和目标入口匹配的公共优化，不包含模型专用优化。

### 使用自定义配置

通过 `--optimization-config` 和 `--runtime-config` 指定自定义的[优化配置](docs/zh/user_guide/optimization_config.md)和[运行配置](docs/zh/user_guide/runtime_config.md)：

```bash
turbo-physai run \
  --optimization-config ./optimization.yaml \
  --runtime-config ./runtime.yaml \
  -- \
  python tools/train.py path/to/config.py
```

### 临时调整 Optimization Group

启动训练时，可通过 `--force-group` 临时放行一个或多个 Group 的可放行检查，通过 `--disable-group` 临时禁用一个或多个 Group。以下命令以 BEVFormer 为例：

```bash
turbo-physai run \
  --model bevformer \
  --force-group bevformer.mdc bevformer.msda \
  --disable-group bevformer.compile bevformer.grid_mask \
  -- \
  torchrun --nproc-per-node=8 tools/train.py \
    ./projects/configs/bevformer/bevformer_base.py \
    --launcher pytorch
```

这两个参数只影响本次运行，不修改 OptimizationConfig。`--force-group` 只能覆盖框架标记为可放行的检查；结构性错误仍会阻断。禁用 Group 后，依赖该 Group 的其他 Group 同时跳过。执行决策及原因记录在 OptimizationReport 中。

### 高级用法：Python API

能够修改训练入口时，可以在导入模型相关模块之前调用一次 `apply()`。无参数调用应用默认公共优化：

```python
import turbo_physai

turbo_physai.apply()
```

指定已支持的模型时，自动选择随包交付的模型优化配置：

```python
import turbo_physai

turbo_physai.apply(model="bevformer")
```

指定自定义优化配置：

```python
import turbo_physai

turbo_physai.apply(
    optimization_config_path="./optimization.yaml",
)
```

临时放行或禁用一个或多个 Optimization Group：

```python
import turbo_physai

turbo_physai.apply(
    model="bevformer",
    force_groups=["bevformer.mdc", "bevformer.msda"],
    disable_groups=["bevformer.compile", "bevformer.grid_mask"],
)
```

`force_groups` 与 `disable_groups` 的约束和报告行为与 Runner 参数一致。

环境准备和完整操作见[安装指南](docs/zh/get_started/installation.md)与[快速开始](docs/zh/get_started/quick_start.md)。具体配置和训练命令见[模型应用说明](model_examples/README.md)。

## 支持模型

| 模型      | 接入基线   | 使用说明                                                |
| --------- | ---------- | ------------------------------------------------------- |
| BEVFormer | 66b65f3... | [BEVFormer 应用说明](model_examples/BEVFormer/README.md) |
| BEVFusion | 326653d... | [BEVFusion 应用说明](model_examples/BEVFusion/README.md) |

完整支持范围见[模型支持清单](docs/zh/models/support_list.md)。

## 仓库结构

```text
TurboPhysAI/
├── turbo_physai/
│   ├── engine/              # 配置解析、检查、执行、恢复和报告
│   ├── operators/           # 可直接调用的算子级 Python API
│   └── optimizations/
│       ├── common/          # 公共算子与框架优化
│       └── models/          # 模型专用优化及随包配置
├── kernel/                  # HCU 原生算子源码
├── model_examples/          # 支持模型的应用说明
├── docs/                    # 用户、开发与设计文档
├── test/                    # 单元测试和框架测试
└── setup.py                 # 构建与安装入口
```

## 文档导航

| 用户目标           | 文档入口                                                                                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 安装并首次运行     | [安装指南](docs/zh/get_started/installation.md) · [快速开始](docs/zh/get_started/quick_start.md)                                                                                   |
| 使用模型优化配置   | [OptimizationConfig 使用指南](docs/zh/user_guide/optimization_config.md) · [RuntimeConfig 使用指南](docs/zh/user_guide/runtime_config.md) · [模型应用说明](model_examples/README.md) |
| 查看结果或排查问题 | [执行报告](docs/zh/user_guide/report.md) · [问题排查](docs/zh/user_guide/troubleshooting.md) · [FAQ](docs/zh/faq.md)                                                                 |
| 开发新的优化       | [优化接入流程](docs/zh/developer_guide/optimization_workflow.md) · [优化声明与 Group 设计](docs/zh/developer_guide/optimization_declarations.md) · [验证清单](docs/zh/developer_guide/validation.md)     |
| 了解实现原理       | [框架架构](docs/zh/design/architecture.md) · [执行与恢复](docs/zh/design/execution.md) · [兼容性与共存](docs/zh/design/compatibility.md)                                             |
| 查询命令参数       | [CLI 参考](docs/zh/reference/cli.md)                                                                                                                                             |

完整文档目录见[文档中心](docs/README.md)。

## 参与开发

优化开发者可以先在仓库外创建独立开发工程：

```bash
turbo-physai optimization init my_model \
  --output ./my_model_optimization
```

完成实现、正确性测试和真实模型验证后，再提交长期集成。开发流程见[优化接入流程](docs/zh/developer_guide/optimization_workflow.md)。

- 贡献要求：[CONTRIBUTING.md](CONTRIBUTING.md)
- 版本记录：[RELEASE_NOTES.md](RELEASE_NOTES.md)
- 安全问题：[SECURITY.md](SECURITY.md)
- 问题和需求：通过仓库 Issue 或 Merge Request 反馈

## License

除第三方和上游派生代码外，TurboPhysAI 自研部分采用 [BSD-3-Clause License](LICENSE)。

本仓库包含经修改的上游开源实现；这些文件继续受其原始许可证、版权和署名要求约束。完整来源、固定基线和第三方许可证材料见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[NOTICE](NOTICE) 和 [third_party/licenses](third_party/licenses)。
