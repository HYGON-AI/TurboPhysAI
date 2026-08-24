# 兼容性与第三方组件共存

本文说明 TurboPhysAI 判断优化适用性的依据，以及与第三方源码修改、运行时替换和启动配置共存时的处理规则。

## 兼容性检查依据

TurboPhysAI 在应用 Optimization Group 前检查以下信息：

- 模型仓库 commit、工作区状态和 repository 标识；
- Python 依赖版本和运行 backend；
- target 的 source Hash、AST Hash，或原生扩展的 artifact Hash；
- target、Alias 与 Replacement 的对象身份、类型和签名；
- Group 声明的 `compatibility_check` 结果。

OptimizationConfig 中的 target Hash 用于确认目标实现是否与生成配置时一致。模型仓库 commit 用于标识和追踪验证基线；commit 不匹配时生成 `warning`，不会单独阻断 Group。需要约束完整仓库状态、依赖版本或其他实现条件时，由 Optimization Group 通过 `compatibility_check` 声明附加检查。

## Alias 处理

Python 的 `from ... import ...` 会在当前模块中保存对象引用：

```python
from framework.ops import original_op
```

此后修改 `framework.ops.original_op`，不会同步改变当前模块已经保存的 `original_op`。因此，优化声明必须将已知的提前导入路径和再次导出路径列入 `aliases`。应用和回滚时，target 与所有 Alias 作为同一替换单元处理。

框架不遍历 `sys.modules` 推断 Alias。自动扫描无法可靠区分需要同步替换的公开入口与模块内部临时引用。

## 第三方修改处理

第三方组件可能通过修改源码、运行时属性赋值、Wrapper、Operator 注册等方式改变模型行为。TurboPhysAI 不判断修改来源，而是检查应用时解析到的实际对象。实际对象与 OptimizationConfig 中的目标证据不一致时，对应 Group 默认被 `block`。

### 同一 target 被多次修改

第三方组件与 TurboPhysAI 修改同一 target 时，按以下规则处理：

- target 证据一致：继续执行其余检查；
- target 证据不一致：默认阻断 Group；
- 仅当失败项被框架标记为可放行时，才可通过 `force_groups` 显式覆盖该检查；
- target 不存在、对象结构冲突、签名不兼容和自定义兼容性检查失败等不可放行。

正式交付应固定组件版本和加载顺序，并基于最终组合重新生成 OptimizationConfig、验证数值与性能。`force_groups` 用于已审查对象的显式覆盖，不替代组合验证。

### Wrapper 组合

当 target 已被第三方 Wrapper 包装时，TurboPhysAI 的 `wrap` 以当前对象为输入继续包装：

```text
原实现 → 第三方 Wrapper → TurboPhysAI Wrapper
```

TurboPhysAI 不从磁盘重新加载原实现，也不沿 `__wrapped__` 链移除已有 Wrapper。因此，最终调用链由各组件的实际应用顺序决定。

### 应用顺序

同一 target 存在多个运行时修改时，最后一次赋值决定该入口最终指向的对象。需要 TurboPhysAI 覆盖第三方实现时，必须在第三方修改完成后调用 `turbo_physai.apply()`，并保证后续初始化不再修改该入口。

默认的 `turbo-physai run` 在每个训练 rank 中先应用 OptimizationConfig，再执行 Python 训练入口。如果第三方组件在训练入口执行期间再次修改同一 target，其修改发生在 TurboPhysAI 之后，框架无法自动恢复 TurboPhysAI 实现。此类组合需要通过固定初始化顺序或专用启动入口处理。

`turbo-physai run` 支持 Python、Torchrun 和 `torchpack dist-run ... python ...` 启动形式。启动器参数由对应适配器转发。包含 `source`、`ulimit`、管道、条件语句或 `torchrun --no-python` 的 Shell 流程仍由外部启动脚本编排，并由脚本调用 `turbo-physai run`。

## 启动环境配置

必须在 Python 启动、`import torch`、通信初始化或设备上下文创建前生效的环境变量，属于 RuntimeConfig 的职责。OptimizationConfig 只描述优化选择与目标证据，`turbo_physai.apply()` 不设置或恢复启动环境。

RuntimeConfig 提供以下配置：

- `environment.set`：设置环境变量；
- `environment.unset`：移除环境变量；
- `process.rank_affinity`：按 rank 设置 CPU 亲和性；
- `process.rank_numa`：按 rank 指定 NUMA 节点；
- `process.numa: auto`：根据可见设备和设备拓扑自动选择 NUMA 节点。

启用自动 NUMA 绑定后，每个 rank 的 Runner 根据 `LOCAL_RANK`、`HIP_VISIBLE_DEVICES` 和 `hy-smi --showtopo` 解析本地 NUMA 节点，并在应用 OptimizationConfig 前通过 `numactl` 重新执行自身。镜像准备、资源限制和复杂 Shell 控制仍由镜像、作业系统或启动脚本负责。

## 能力边界

TurboPhysAI 不自动判断以下事项：

- 两段实现是否语义等价；
- 模型函数是否已内联某项公共优化；
- 第三方修改来自磁盘源码还是运行时替换；
- 动态库或 Operator 注册是否能够安全撤销；
- 未执行的动态分支是否满足正确性和性能要求。

上述事项需要通过代码审查、固定加载顺序以及真实模型的数值、性能和稳定性测试进行验证。
