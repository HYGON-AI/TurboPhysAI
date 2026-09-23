# 优化验证与交付

本指南规定 Optimization Group 从开发完成到交付使用前需要完成的验证。最终交付只保留具有可复现端到端性能收益的优化，并要求优化后的模型精度在统一评测口径和既定容差内与基线一致。

OptimizationReport 中的 `applied` 仅表示 Group 已成功安装到当前训练进程，不能替代数值正确性、模型精度和性能验证。

## 1. 验证层级

| 验证层级 | 适用范围 | 主要结论 |
|---|---|---|
| 接入检查 | 所有 Optimization Group | 在目标环境中，预期 Group 成功应用，无非预期的跳过、阻断或失败 |
| 正确性验证 | 新增或修改的优化实现 | 计算结果或行为符合预期；涉及计算改写时，输出及必要梯度与参考实现在容差内一致 |
| 模型验证 | 模型专用优化和完整 OptimizationConfig | 目标训练流程正常，模型精度满足验收标准 |
| 性能验证 | 拟交付的性能优化 | 在一致评测口径下取得可复现的端到端性能收益 |

各类验证按优化实际影响范围执行。仅为性能优化提供必要支撑的兼容 Group 可以不单独计算性能收益，但必须证明其为完整优化配置的必要依赖，且不会引入精度或性能回归。

## 2. Optimization Group 接入检查

开发者在目标模型环境中使用交付配置运行，检查 OptimizationReport，确认预期 Group 成功应用，无非预期的跳过、阻断或失败。

声明校验、应用、阻断、回滚、报告和异常传播等通用机制由 `test/engine/` 下的公共测试统一覆盖，优化接入无需逐项重复编写测试。

[`test/engine/test_generated_configs.py`](../../../test/engine/test_generated_configs.py) 自动检查公共优化和模型专用优化的 Config、Recipe、Catalog 及继承配置是否与生成记录一致，用于发现修改后未重新生成配置等误操作。该检查无需导入模型，不验证优化计算或模型运行结果。

## 3. 正确性验证

开发者为新增或修改的优化实现提供单元测试。测试优先使用合成输入和最小依赖，可按需使用 HCU。必须依赖完整模型环境的验证，由开发者在接入阶段完成并提供证据。

| 改动类型 | 验证内容 |
|---|---|
| 计算逻辑改写 | 输出及必要梯度与参考实现的数值对比，以及形状、数据类型和返回值等调用契约 |
| 纯 compile 包装、参数配置 | 包装目标、参数、开关和调用行为 |
| 数据预取与处理 | 数据内容、顺序、完整性，以及设备同步、迭代结束和异常退出时的资源管理 |
| 自定义适用条件 | `compatibility_check` 的兼容与不兼容场景，以及 `runtime_condition` 条件成立与不成立时的判断和执行分支 |
| 自定义导入兼容逻辑 | 需要应用、环境已兼容、存在冲突三种场景下的处理结果 |

Wrapper 应保持原目标的外部调用契约。运行条件允许使用优化实现时，验证该实现的计算结果；不满足条件时，确认调用原实现。

数值测试应固定随机种子、权重和输入，在相同设备、数据类型和输入布局下比较参考实现与优化实现。容差根据算子精度和数据类型确定，并在测试中明确记录。

数值测试示例：

- [`test/test_grid_sample.py`](../../../test/test_grid_sample.py) 使用 PyTorch `grid_sample` 作为参考实现，对比 TurboPhysAI 算子的前向输出、输入梯度和 Grid 梯度；
- [`test/optimizations/test_hcu_implementations.py`](../../../test/optimizations/test_hcu_implementations.py) 对比公共与 BEVFormer LightOp Multi-Scale Deformable Attention 的前向与反向结果；
- [`test/test_deformable_aggregation.py`](../../../test/test_deformable_aggregation.py) 使用参考计算验证 Deformable Aggregation 的输出和梯度；
- [`test/engine/test_bevfusion.py`](../../../test/engine/test_bevfusion.py) 中的 `test_factorized_depth_features_match_dense_outer_product_and_gradients` 对比 BEVFusion 稠密参考计算与优化实现，并验证输入梯度。

训练算子应验证模型实际依赖的全部梯度。明确不需要梯度的输入，可以不做梯度对比，但必须在实现约束和测试中说明。输出形状、数据类型、设备位置以及返回值数量也属于调用契约，应一并验证。

原生算子还应覆盖声明支持的 shape、dtype、布局和边界输入。

## 4. 模型训练验证

开发者在目标模型环境中，使用完整交付 OptimizationConfig 完成目标训练流程并验证模型精度。开发排查时可单独启用 Group 或逐步组合；存在依赖的 Group 按必要组合验证。

完整 OptimizationConfig 的精度应与未启用 TurboPhysAI 的模型基线对比。精度一致是指在相同数据集、评测脚本和评测配置下，关键指标满足项目规定的验收容差，不要求训练过程中的每个浮点结果完全相同。

验证记录包括训练完成情况、基线与优化后的模型评测指标，以及交付范围内的单卡或多卡运行结果。根据优化影响补充 loss、梯度、NaN/Inf 或稳定性记录。

模型级测试命令、数据准备、权重、配置、精度和性能结果由对应 `model_examples/<Model>/README.md` 维护。

交付范围包含多卡训练时，应验证实际交付的启动方式、进程数、环境配置和退出行为。使用 `torch.compile` Wrapper 时，应覆盖交付范围内的训练模式、输入形状和编译配置。

## 5. 性能验证

基线与优化结果应采用一致的任务和评测口径，记录测试环境、配置差异、测试方法和结果，确保结果可复现。

性能以完整模型的端到端耗时或吞吐量为主要指标，同时关注显存占用。涉及编译时，首次编译开销与稳态性能分开统计。

交付配置应具有可复现的性能收益，并满足模型精度要求。
