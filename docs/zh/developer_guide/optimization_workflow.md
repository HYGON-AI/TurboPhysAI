# 优化开发与接入流程

本文面向开发模型或后端算子优化的人员。新优化可先在外部工程中独立开发和验证；准备合入 TurboPhysAI 时，再迁移至公共优化或模型专用优化目录。

## 开发主线

```text
准备基线 → 创建外部工程 → 优化声明 → 生成 OptimizationConfig → 验证 → 提交评审
```

TurboPhysAI 通过 Python 运行时替换，将 Replacement 安装到 Catalog 声明的目标入口，无需修改模型源码。框架在替换前检查目标代码和优化组合，并以 Optimization Group 为单位执行和回滚。具体机制见[组件架构](../design/architecture.md)。

## 1. 准备基线

开始开发前，应在未启用 TurboPhysAI 优化的条件下完成一次可复现的基线运行，并记录：

- 模型仓库、接入 commit 和工作区状态；
- 硬件、软件版本、训练配置和启动命令；
- 数据集、预训练权重及其准备方式；
- 基线精度、性能、显存和稳定性结果；
- 待优化入口的参数、返回值、Tensor 形状、数据类型、设备及梯度契约。

后续数值、训练和性能验证应使用同一基线与测试条件。基线确认后，再创建与模型源码分离的优化开发工程。

## 2. 创建外部工程

使用 `optimization init` 在 TurboPhysAI 仓库外创建独立的优化开发与测试工程：

```bash
turbo-physai optimization init customer_model \
  --output ./customer_model_optimization
cd customer_model_optimization
python -m pip install -e .
```

生成目录：

```text
customer_model_optimization/
├── README.md                         # 开发说明
├── pyproject.toml                    # Python 包配置
├── customer_model_optimization/
│   ├── __init__.py
│   ├── catalog.py                    # 声明 Optimization Group
│   └── replacements.py               # 实现 Replacement
├── configs/
│   └── recipe.yaml                   # 选择生成配置所需的 Group
└── tests/
    └── test_catalog.py               # Catalog 导入与优化单元测试
```

该工程分别保存优化实现、优化声明、配置配方和测试。完成初始化后，即可根据第一步确定的目标入口实现优化。

## 3. 优化声明

在 `replacements.py` 中实现优化对象，并保持原目标的参数、返回值、Tensor 形状、数据类型、设备和梯度契约。在 `catalog.py` 中使用 `replace` 或 `wrap` 关联目标入口与 Replacement，再使用 `group` 将共同构成一项完整优化的成员组织为 Optimization Group。

每个 Group 应具有清晰的功能边界，并能够独立检查、应用和回滚。目标对象存在其他导入路径时声明 Alias；优化只适用于部分运行时输入时声明运行条件。实现和声明完成后，应在 `tests/` 中增加对应的导入、数值及必要的梯度测试。

接口、参数和真实示例见[优化声明](optimization_declarations.md)；涉及原生算子时参见[自定义算子接入](custom_operator.md)。完成声明后，将需要交付的 Group ID 写入 `configs/recipe.yaml`，进入配置生成阶段。

## 4. 生成 OptimizationConfig

Recipe 记录需要启用的 Group，并可继承公共基础优化。在干净的模型仓库和确定的优化接入 commit 上执行：

```bash
turbo-physai optimization generate \
  --recipe configs/recipe.yaml \
  --repo /path/to/model/repository \
  --commit <validated_commit> \
  --output configs/optimization.yaml
```

生成过程会加载 Catalog，展开 Group 依赖，检查目标冲突和 Replacement 引用，并将目标源码证据写入 OptimizationConfig。模型优化需要固定环境变量时，由开发者根据验证结果在同一 `configs/` 目录编写 RuntimeConfig；不存在额外启动要求时无需创建该文件。

配置生成规则见[生成和检查 OptimizationConfig](optimization_config_generation.md)，运行环境字段见[RuntimeConfig 使用指南](../user_guide/runtime_config.md)。生成的 OptimizationConfig 与第二步建立的测试工程共同进入验证阶段。

## 5. 验证

先执行 Replacement 单元测试，再分别验证单个 Group 和完整 OptimizationConfig。切换到模型仓库，使用外部工程生成的配置启动原训练入口：

```bash
cd /path/to/model/repository
turbo-physai run \
  --optimization-config \
    /path/to/customer_model_optimization/configs/optimization.yaml \
  --runtime-config \
    /path/to/customer_model_optimization/configs/runtime.yaml \
  python tools/train.py <原训练参数>
```

未使用 RuntimeConfig 时，删除对应参数。数值、梯度、训练稳定性、精度、性能和显存应与第一步记录的基线在相同条件下比较。验证过程形成测试结果、训练记录、性能数据和 OptimizationReport，作为后续评审依据。

验收要求见[优化验证与交付](validation.md)，报告字段见[优化应用报告](../user_guide/report.md)。

## 6. 提交评审

提交内容包括 Replacement、Catalog、Recipe、生成后的配置、接入 commit、测试结果、训练与性能记录、OptimizationReport 以及必要的使用说明。

与具体模型上下文无关且入口稳定的能力迁入 `turbo_physai/optimizations/common/`；依赖模型结构、数据链路或训练过程的能力迁入 `turbo_physai/optimizations/models/<model>/`。合入时同步维护测试、模型应用说明和支持清单。

提交和评审要求见[贡献指南](../../../CONTRIBUTING.md)。
