# 组件架构

TurboPhysAI 通过 Python 运行时替换（Monkey Patch），在不修改模型源码的情况下应用性能优化。为简化优化接入，并使替换过程可检查、可回滚、可追溯，组件统一管理优化声明与配置、应用条件检查、冲突与执行顺序、分组执行与回滚以及状态记录，形成如下架构。TurboPhysAI 基于 Python 运行时替换（Monkey Patch）应用优化。为降低优化接入成本，并保证替换过程可检查、可回滚、可追溯，组件围绕优化声明、配置管理、应用前检查、冲突分析、分组执行和结果记录构建了如下架构。

![TurboPhysAI 组件架构](../../assets/turbophysai-component-framework.png)

## 分层结构

架构中的四层从上到下分别回答：如何启用优化、应用哪些优化、如何安全地完成应用，以及优化代码由什么实现。训练从命令或 Python API 进入组件，配置确定本次需要使用的优化，优化应用层完成检查和安装，模型随后通过原有入口调用优化实现。

### 命令与接口层

这一层是训练用户和优化开发者使用 TurboPhysAI 的入口。

- `turbo-physai run` 负责在不修改模型源码的情况下应用优化，并将 RuntimeConfig 中声明的环境变量配置到训练进程；
- `turbo-physai optimization generate` 将优化声明和模型基线转换为可交付的优化配置；
- `turbo_physai.apply()` 用于在 Python 训练入口中显式应用优化。

训练用户通常使用 `turbo-physai run`。Python API 适用于能够控制训练入口的集成方式，配置生成命令用于优化开发与交付。

### 声明与配置层

这一层描述“有哪些优化”和“本次训练使用哪些优化”：

- **Catalog** 是可用优化的清单，记录每项优化需要替换的模型入口及其优化实现；
- **OptimizationConfig** 从 Catalog 中选择本次需要应用的 Optimization Group，并保存用于确认目标代码是否匹配的证据；
- **RuntimeConfig** 记录训练启动所需的环境变量和进程资源配置。

模型专用配置随组件交付；使用 `--model` 启动训练时，TurboPhysAI 自动加载对应的 OptimizationConfig 和 RuntimeConfig。

### 优化应用层

这一层负责将配置中的优化安全地安装到当前训练进程。它先定位模型中的目标对象，确认目标代码和运行环境满足应用条件，再处理多项优化之间的依赖和冲突，最后按照确定的顺序安装优化。

安装过程中，框架以 Optimization Group 为单位保存目标原始状态。某个 Group 安装失败时，框架恢复该 Group 已修改的目标；检查结论、应用状态和失败原因会统一记录并输出。

### 优化实现层

这一层提供最终参与模型计算的优化代码，包括：

- 可被多个模型复用的公共优化；
- 针对具体模型计算图和执行链路实现的模型专用优化；
- 由 HCU Kernel、hipDNN、LightOp 或 PyTorch 扩展提供的高性能算子。

优化应用完成后，模型仍然使用原有调用入口，实际执行的是已经安装的优化实现。

## 工程目录

```text
turbo_physai/
├── kernel/                         # 原生算子源码
├── turbo_physai/
│   ├── csrc/                       # 统一 PyBind 入口
│   ├── operators/                  # 可直接调用的算子 Python API
│   ├── optimizations/
│   │   ├── common/                 # 跨模型公共优化
│   │   │   ├── catalog.py          # 公共优化登记入口
│   │   │   ├── configs/            # 默认公共 OptimizationConfig
│   │   │   ├── mmcv/
│   │   │   │   ├── catalog.py
│   │   │   │   └── configs/
│   │   │   └── mmdet3d/
│   │   │       ├── catalog.py
│   │   │       └── configs/
│   │   └── models/                 # 模型专用优化
│   │       ├── bevformer/
│   │       │   ├── catalog.py
│   │       │   └── configs/
│   │       └── bevfusion/
│   │           ├── catalog.py
│   │           └── configs/
│   ├── engine/
│   │   ├── definitions/            # Group 和 Replacement 声明、登记
│   │   ├── config/                 # OptimizationConfig 加载、校验和生成
│   │   ├── checking/               # 环境、证据、依赖与冲突检查
│   │   ├── execution/              # Replacement、快照、回滚和报告
│   │   ├── contracts.py            # 各阶段共享的数据结构
│   │   └── errors.py               # Engine 异常
│   ├── development/                # 外部优化工程初始化
│   ├── compatibility.py            # 兼容性补丁声明接口
│   ├── bootstrap/                  # 解释器启动钩子，在 rank 内自动激活
│   ├── runner.py                   # Python 兼容入口与进程资源绑定辅助
│   ├── runtime.py                  # RuntimeConfig 加载与进程环境准备
│   └── cli.py                      # 命令行入口
├── model_examples/
│   ├── BEVFormer/                  # BEVFormer 应用说明
│   └── BEVFusion/                  # BEVFusion 应用说明
├── docs/                            # 对外文档
├── scripts/                         # 文档检查和工程辅助脚本
├── test/                            # 算子、Engine 和 Runner 测试
├── third_party/                     # 第三方许可证材料
└── setup.py                         # Python 包和原生扩展构建
```
