# 优化验证与交付

本指南规定 Optimization Group 从开发完成到交付使用前需要完成的验证。最终交付只保留具有可复现端到端性能收益的优化，并要求优化后的模型精度在统一评测口径和既定容差内与基线一致。

OptimizationReport 中的 `applied` 仅表示 Group 已成功安装到当前训练进程，不能替代数值、梯度、模型精度和性能验证。

## 1. 验证层级

| 验证层级 | 适用范围 | 主要结论 |
|---|---|---|
| 接入检查 | 所有 Optimization Group | 声明可解析，检查、应用、报告和回滚流程正确 |
| 数值验证 | 改变计算结果的优化 | 输出与参考实现一致，数据类型、设备和形状符合调用契约 |
| 梯度验证 | 参与训练反向传播的优化 | 模型所需梯度与参考实现一致 |
| 模型验证 | 模型专用优化和完整 OptimizationConfig | 训练可运行，损失与精度符合验收标准 |
| 性能验证 | 所有拟交付的性能优化 | 在统一测试条件下获得可复现的端到端稳态收益 |
| 专项验证 | 使用条件分发、编译、导入兼容或分布式能力的优化 | 对应能力的适用路径和边界均已覆盖 |

前五类验证按优化实际影响范围执行；专项验证仅在 Optimization Group 使用相应能力时执行。仅为性能优化提供必要支撑的兼容 Group 可以不单独计算性能收益，但必须证明其为完整优化配置的必要依赖，且不会引入精度或性能回归。

## 2. Optimization Group 接入检查

每个 Optimization Group 至少需要验证：

- Catalog 可以加载，Group ID 和成员 ID 不重复；
- `target`、Replacement 和 `aliases` 均可解析；
- `turbo_physai.check()` 只完成检查，不安装 Replacement；
- Group 可以独立应用，OptimizationReport 中记录正确的规划决策和执行结果；
- 任一成员应用失败时，已经修改的成员能够按 Group 边界恢复；
- Replacement 抛出的 Python 异常保留原始 Traceback，并能定位到具体优化实现。

使用附加能力时，还需要完成以下检查：

- 声明 `compatibility_check`：分别覆盖兼容和不兼容环境，确认不兼容时在应用前阻断该 Group；
- 声明 `runtime_condition`：分别构造条件成立和不成立的输入，验证优化实现与原实现的输出及必要梯度；条件函数异常必须直接暴露；
- 使用 `wrap`：验证 Wrapper 返回可调用对象，并保持原目标的外部调用契约；
- 使用导入兼容接口：分别验证“需要应用”“环境已兼容”和“存在冲突”三种结果。

Engine 的声明、检查、事务执行和报告行为由 `test/engine/` 下的测试统一覆盖。模型优化应补充真实 Catalog 和 Group 的接入测试。

## 3. 数值与梯度验证

数值测试应固定随机种子、权重和输入，在相同设备、数据类型和输入布局下比较参考实现与优化实现。容差根据算子精度和数据类型确定，并在测试中明确记录，不应仅为通过测试而放宽。

仓库中的真实测试可作为实现参考：

- [`test/test_grid_sample.py`](../../../test/test_grid_sample.py) 使用 PyTorch `grid_sample` 作为参考实现，对比 TurboPhysAI 算子的前向输出、输入梯度和 Grid 梯度；
- [`test/test_multi_scale_deformable_attn.py`](../../../test/test_multi_scale_deformable_attn.py) 对比 Multi-Scale Deformable Attention 的前向与反向结果；
- [`test/test_deformable_aggregation.py`](../../../test/test_deformable_aggregation.py) 使用参考计算验证 Deformable Aggregation 的输出和梯度；
- [`test/engine/test_bevfusion.py`](../../../test/engine/test_bevfusion.py) 中的 `test_factorized_depth_features_match_dense_outer_product_and_gradients` 对比 BEVFusion 稠密参考计算与优化实现，并验证输入梯度。

训练算子应验证模型实际依赖的全部梯度。明确不需要梯度的输入，可以不做梯度对比，但必须在实现约束和测试中说明。输出形状、数据类型、设备位置以及返回值数量也属于调用契约，应一并验证。

## 4. 专项验证

以下项目根据 Optimization Group 使用的能力执行：

| 能力 | 验证要求 |
|---|---|
| 原生算子 | 覆盖声明支持的 shape、dtype、布局和边界输入；训练算子同时验证 Backward |
| `runtime_condition` | 分别验证条件成立与不成立时的输出、必要梯度和异常传播 |
| `torch.compile` Wrapper | 覆盖交付范围内的训练模式、输入形状和编译配置，分别记录首次编译与稳态运行结果 |
| 导入兼容接口 | 验证依赖缺失、环境已兼容和名称冲突时的检查与应用结果 |
| 多卡启动 | 验证实际交付的启动器、进程数、环境配置和退出行为 |

OptimizationReport 只能证明条件分发器、Wrapper 或导入兼容动作已经安装，不能证明所有运行路径都已被训练数据触发。因此，专项能力的业务路径需要通过独立测试或模型训练记录确认。

## 5. 模型训练验证

模型优化按以下顺序验证：

1. 单独启用每个 Optimization Group，确认功能边界和失败定位准确；
2. 按依赖关系逐步组合 Group，排除组合导致的数值或性能回归；
3. 使用完整 OptimizationConfig 完成目标训练流程。

完整 OptimizationConfig 的精度应与未启用 TurboPhysAI 的模型基线对比。精度一致是指在相同数据集、评测脚本和评测配置下，关键指标满足项目规定的验收容差，不要求训练过程中的每个浮点结果完全相同。

训练记录至少包括：

- 首次有效 loss；
- 固定区间内的 loss 均值、趋势和关键分量；
- 代表性参数梯度；
- NaN/Inf 统计；
- 模型约定的精度或评测指标；
- 单卡冒烟结果，以及交付范围包含多卡时的多卡和稳定性结果。

模型级测试命令、数据准备、权重、配置、精度和性能结果由对应 `model_examples/<Model>/README.md` 维护，通用验证规则不重复保存模型数据。

## 6. 性能验证

性能对比必须保证基线与优化实现使用相同输入序列、Batch Size、进程数、资源绑定和运行时配置。测试记录至少包括：

- 设备型号、卡数、Batch Size 和 DataLoader workers；
- Python、PyTorch、训练框架和 TurboPhysAI 版本；
- OptimizationConfig、RuntimeConfig 和模型基线；
- 是否启用 Profiler、编译模式和缓存状态；
- 预热区间、统计区间、平均值、中位数、P90 和显存占用。

存在编译过程时，应将首次编译开销与稳态迭代分开统计。仓库算子测试使用 [`test/utils.py`](../../../test/utils.py) 中的统一预热和计时工具，新增算子应优先复用同一测试口径。

性能结论以完整模型的端到端训练结果为准。未获得可复现正收益、收益无法排除测试波动，或导致其他关键性能指标明显回归的优化，不纳入最终交付配置。

## 7. 交付验收

所有优化必须满足：

- [ ] 官方模型基线和 commit 已记录；
- [ ] OptimizationConfig 已通过 `turbo-physai optimization validate`；
- [ ] 在目标模型仓库中完成 `turbo-physai optimization check`；
- [ ] Optimization Group 边界和依赖关系已经评审；
- [ ] 适用的数值与梯度测试通过；
- [ ] 完整模型的关键精度指标在既定容差内与基线一致；
- [ ] 完整模型获得可复现的端到端性能收益；
- [ ] OptimizationReport 中不存在未解释的 `blocked`、`failed` 或 `rolled_back`；
- [ ] 环境、启动方式和已知限制已写入对应模型 README。

包含专项能力时，还必须满足：

- [ ] 编译、条件分发、导入兼容等适用路径均已覆盖；
- [ ] 交付范围包含多卡或长时间训练时，相应稳定性验证已经完成。
