# Python API 参考

Python API 适用于能够控制训练入口和模型模块导入顺序的集成方式。常规多进程训练优先使用 [`turbo-physai run`](cli.md#run)，由 Runner 在每个训练进程中应用优化并准备 RuntimeConfig。

## `turbo_physai.apply()`

`apply()` 加载一份 OptimizationConfig，检查其适用性，在当前 Python 进程中安装满足条件的优化，并返回本次应用报告。

```python
turbo_physai.apply(
    *,
    optimization_config_path=None,
    model=None,
    log_report=False,
    registry=None,
    catalog=None,
    force_groups=(),
    disable_groups=(),
) -> OptimizationReport
```

### 基本用法

在导入目标模型模块前调用 `apply()`：

```python
import turbo_physai

report = turbo_physai.apply(
    model="bevformer",
    log_report=True,
)

# 在 apply() 返回后导入并执行目标模型代码。
```

`model` 的可选值见[模型支持清单](../models/support_list.md)。省略 `model` 和 `optimization_config_path` 时，组件按配置选择顺序加载默认 OptimizationConfig。

加载外部 OptimizationConfig：

```python
import turbo_physai

report = turbo_physai.apply(
    optimization_config_path="./configs/optimization.yaml",
    log_report=True,
)
```

### 参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `optimization_config_path` | `str` 或 `os.PathLike` | `None` | 显式 OptimizationConfig 路径；优先于 `model` 和其他配置来源 |
| `model` | `str` | `None` | 随包交付的内置模型名称；名称不区分大小写，`-` 按 `_` 处理 |
| `log_report` | `bool` | `False` | 为 `True` 时，由 Rank 0 将完整 OptimizationReport 输出到标准日志 |
| `force_groups` | 字符串序列 | `()` | 临时放行指定且已启用 Group 的可放行检查；结构性错误不能放行 |
| `disable_groups` | 字符串序列 | `()` | 本次调用不应用指定 Group；依赖这些 Group 的其他 Group 同时跳过 |
| `registry` | `Registry` | `None` | 自定义优化声明注册表；主要用于外部优化包、扩展和测试 |
| `catalog` | `OptimizationConfigCatalog` | `None` | 自定义配置目录，用于解析 OptimizationConfig 的 `extends` |

`force_groups` 和 `disable_groups` 接收 Group ID 序列，不能直接传入单个字符串。同一 Group 不能同时出现在两个参数中。

### 配置选择顺序

`apply()` 按以下顺序选择 OptimizationConfig：

1. `optimization_config_path` 指定的文件；
2. `model` 对应的随包配置；
3. `TURBO_PHYSAI_OPTIMIZATION_CONFIG` 环境变量指定的文件；
4. 当前工作目录下的 `turbophysai_configs/default/optimization.yaml`；
5. 随包交付的公共 OptimizationConfig。

Python API 不加载或应用 RuntimeConfig，也不负责启动训练命令。需要 RuntimeConfig 时，应使用 `turbo-physai run`，或由调用方在执行模型代码前自行准备运行环境。

### 返回值

`apply()` 返回 `OptimizationReport`。常用字段包括：

| 字段 | 说明 |
| --- | --- |
| `run_id` | 本次优化应用的唯一标识 |
| `optimization_config` | 解析后的 OptimizationConfig |
| `prepared_execution` | Group 检查结果、决策、冲突和执行顺序 |
| `execution` | 各 Group 及其 Replacement 的执行结果 |
| `summary` | `applied`、`skipped`、`blocked`、`failed`、`rolled_back` 和 `not_started` 计数 |
| `optimization_config_path` | 本次实际加载的 OptimizationConfig 绝对路径 |
| `runtime_config_path` | Runner 提供的 RuntimeConfig 路径；直接调用 API 且未设置时为 `None` |

应用成功返回不等于全部 Group 均为 `applied`。Group 被阻断或成功回滚时，`apply()` 仍可返回报告。调用方可以检查 Summary：

```python
unavailable = (
    "blocked",
    "failed",
    "rolled_back",
    "not_started",
)

if any(report.summary[name] for name in unavailable):
    raise RuntimeError("not all enabled Optimization Groups were applied")
```

报告结构、状态和日志格式见[优化应用报告](../user_guide/report.md)。

### 调用约束和异常

- 每个 Python 进程只能调用一次 `apply()`；调用失败后也不能在同一进程中重试。
- 应在导入目标模型模块前调用，确保模型代码通过已安装的优化入口执行。
- `force_groups` 只能覆盖框架明确标记为可放行的检查。
- Group 应用失败且回滚成功时，报告将该 Group 记为 `rolled_back`，其他独立 Group 可以继续执行。
- 回滚失败时抛出 `OptimizationRollbackError`，异常的 `report` 属性保存已生成的报告。
- 配置不存在或配置内容无效时，分别抛出 `OptimizationConfigNotFoundError` 或 `OptimizationConfigError`。

## `turbo_physai.check()`

`check()` 加载 OptimizationConfig 并计算 Group 决策，但不安装 Replacement：

```python
prepared = turbo_physai.check(
    model="bevformer",
    disable_groups=("bevformer.compile.encoder",),
)
```

```python
turbo_physai.check(
    *,
    optimization_config_path=None,
    model=None,
    registry=None,
    catalog=None,
    force_groups=(),
    disable_groups=(),
) -> PreparedExecution
```

返回值为 `PreparedExecution`，包含运行环境快照、Group 检查结果、应用决策、冲突和执行顺序。

检查过程可能导入 target 和 Replacement 模块，`wrap()` 声明还会调用 Wrapper 构造函数。框架会尽力恢复 `sys.modules`，但不能撤销模块导入产生的任意外部副作用。因此，`check()` 适用于开发和问题排查，不替代独立进程中的正式训练验证。
