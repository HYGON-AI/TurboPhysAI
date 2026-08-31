# TurboPhysAI



TurboPhysAI 是面向 Physical AI 的模型训练性能优化组件，面向自动驾驶、具身智能等应用领域，并支持世界模型等相关模型的训练优化。组件在**不修改模型源码**的前提下，通过算子、模型计算图、数据链路和训练过程优化，充分发挥 HCU 高性能计算能力。

优化实现与模型源码分离维护：优化能力随包交付，在训练进程启动时应用到目标入口，并记录每项优化的安装状态，便于确认模型执行链路使用的对象并回溯问题。

能力边界、适用场景和整体工作方式见 [TurboPhysAI 概述](docs/zh/overview.md)。

## 使用方式

TurboPhysAI 提供命令行和 Python API 两类入口。训练用户优先使用 `turbo-physai run`；能够控制模型模块导入顺序时，也可以直接调用 `turbo_physai.apply()`。

### 使用内置模型优化

原训练命令保持不变，在前面加上 `turbo-physai run --model <模型名>`，即可加载该模型随包交付的 [OptimizationConfig](docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](docs/zh/user_guide/runtime_config.md)：

```bash
# 原训练命令
torchrun --nproc-per-node=8 tools/train.py \
  ./projects/configs/bevformer/bevformer_base.py --launcher pytorch

# 启用 BEVFormer 的模型专用优化
turbo-physai run --model bevformer \
  torchrun --nproc-per-node=8 tools/train.py \
    ./projects/configs/bevformer/bevformer_base.py --launcher pytorch
```

组件在每个训练进程启动时应用一次优化。增加 `--log-report` 可以输出优化应用日志，用于确认预期优化项是否已应用。`--model` 的可选值、优化接入基线和使用说明见[模型支持清单](docs/zh/models/support_list.md)。

### 仅使用公共优化

不指定模型和配置时，Runner 加载随包交付的公共 OptimizationConfig，不加载模型专用优化或 RuntimeConfig：

```bash
turbo-physai run \
  python tools/train.py path/to/config.py
```

### 使用显式配置

外部优化包或自定义优化时可以显式指定 OptimizationConfig 和 RuntimeConfig：

```bash
turbo-physai run \
  --optimization-config ./configs/optimization.yaml \
  --runtime-config ./configs/runtime.yaml \
  python tools/train.py path/to/config.py
```

显式配置路径优先于 `--model` 选择的同类配置。RuntimeConfig 是可选项；未使用 RuntimeConfig 时，训练环境和启动参数由原命令或外部脚本负责。

### 使用 Python API

能够控制训练入口时，可以在导入模型相关模块前显式应用优化：

```python
import turbo_physai

report = turbo_physai.apply(
    model="bevformer",
    log_report=True,
)
```

也可以将 `model` 替换为 `optimization_config_path="./configs/optimization.yaml"`，加载显式 OptimizationConfig。`apply()` 每个训练进程只能调用一次，不负责加载 RuntimeConfig 或启动训练；调用方需要在应用完成后再导入并执行模型代码。

前置条件（安装组件、模型仓库 commit、数据集与权重）见[安装指南](docs/zh/get_started/installation.md)。完整启动参数、配置规则和报告说明见[快速开始](docs/zh/get_started/quick_start.md)、[CLI 参考](docs/zh/reference/cli.md)和[优化应用报告](docs/zh/user_guide/report.md)。

## 按角色导航

| 角色         | 典型场景                                         | 文档入口                                                                                                                                   |
| ------------ | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 训练用户     | 在已支持的模型上启用优化，确认安装状态并排查问题 | [安装指南](docs/zh/get_started/installation.md) → [快速开始](docs/zh/get_started/quick_start.md) → [模型支持清单](docs/zh/models/support_list.md) |
| 优化开发人员 | 为新模型或新算子实现优化，生成配置并完成验证交付 | [优化开发与接入流程](docs/zh/developer_guide/optimization_workflow.md) → [优化验证与交付](docs/zh/developer_guide/validation.md)                |
| 框架维护人员 | 维护优化引擎、评审优化接入、扩展框架能力         | [组件架构](docs/zh/design/architecture.md) → [优化应用与回滚](docs/zh/design/execution.md) → [兼容性管理](docs/zh/design/compatibility.md)       |

日常查询：[CLI 参考](docs/zh/reference/cli.md) · [优化应用报告](docs/zh/user_guide/report.md) · [问题排查](docs/zh/user_guide/troubleshooting.md) · [FAQ](docs/zh/faq.md)。完整目录见[文档中心](docs/README.md)。

## 仓库结构

```text
TurboPhysAI/
├── turbo_physai/      组件实现：优化引擎、算子 API、随包交付的优化与配置
├── kernel/            HCU 原生算子源码
├── model_examples/    已支持模型的应用说明
├── docs/              用户、开发与设计文档
└── test/              算子与框架测试
```

各层职责和完整工程目录见[组件架构](docs/zh/design/architecture.md)。

## 参与开发

贡献说明见 [CONTRIBUTING.md](CONTRIBUTING.md)。

版本记录：[RELEASE_NOTES.md](RELEASE_NOTES.md) · 安全问题：[SECURITY.md](SECURITY.md) · 问题和需求：通过仓库 Issue 或 Pull Request（PR）反馈。

## License

除第三方和上游派生代码外，TurboPhysAI 自研部分采用 [BSD-3-Clause License](LICENSE)。

本仓库包含经修改的上游开源实现；这些文件继续受其原始许可证、版权和署名要求约束。完整来源、固定基线和第三方许可证材料见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[NOTICE](NOTICE) 和 [third_party/licenses](third_party/licenses)。
