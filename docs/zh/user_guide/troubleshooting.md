# 问题排查

问题排查应先保留完整启动命令、终端输出和 OptimizationReport。报告中的 `Configuration Checks`、`Preparation` 和 `Execution` 分别对应配置级检查、Group 应用决策和实际执行结果。

## 1. 找不到内置模型或配置文件

典型错误：

```text
unknown built-in model ...
OptimizationConfig not found: ...
RuntimeConfig not found: ...
```

处理方法：

1. 使用 `turbo-physai run --help` 确认参数名称；
2. 内置模型使用 `--model <name>`，不要手工拼接安装目录；
3. 自定义交付使用 `--optimization-config` 和 `--runtime-config` 指定文件；
4. 显式路径建议使用绝对路径，或在启动脚本中基于脚本位置解析。

配置选择规则见 [OptimizationConfig](optimization_config.md) 和 [RuntimeConfig](runtime_config.md)。

## 2. Group 未应用

Runner 完成每个 rank 的优化处理后会输出：

```text
TURBO_PHYSAI_OPTIMIZATION_COMPLETED rank=<rank> applied=... blocked=... failed=...
```

如果 `blocked`、`failed`、`rolled_back` 或 `not_started` 不为 0，应查看 OptimizationReport：

- `Preparation` 中的 `block`：应用前条件未满足；
- `Execution` 中的 `rolled_back`：Group 应用失败，但已恢复应用前状态；
- `Execution` 中的 `failed`：快照失败，或应用失败后未能完整恢复；
- `Execution` 中的 `not_started`：依赖 Group 未成功应用，或此前出现终止性错误。

独立 Group 的阻断或成功回滚不会阻止其他 Group 和训练入口继续运行，因此不能仅根据训练是否启动判断全部优化均已生效。

## 3. target、alias 或 Hash 检查失败

常见原因包括：

- 当前模型或依赖代码与生成 OptimizationConfig 时的基线不同；
- target 导入路径已经变化；
- Catalog 未声明目标对象被提前导出后形成的 alias；
- 第三方组件已修改同一 Python 对象；
- Replacement 的调用签名与 target 不兼容。

先执行：

```bash
git status --short
turbo-physai optimization check /path/to/optimization.yaml \
  --repo /path/to/model
```

结合报告中的 Group、Replacement ID、target、`expected` 和 `actual` 定位差异。模型基线正式升级后，应重新完成验证并生成 OptimizationConfig，不应依赖导入顺序形成隐式覆盖。

`project.commit = warning` 表示仓库 HEAD 与配置记录不同，但该差异本身不阻断应用。是否可以应用仍以 target Hash、签名和其他检查结果为准。

## 4. 需要临时放行检查差异

开发人员确认差异安全后，可以仅对本次启动指定 Group：

```bash
turbo-physai run \
  --model bevformer \
  --force-group <group-id> \
  -- \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

`--force-group` 只覆盖框架标记为可放行的检查。target 不存在、alias 指向不同对象、签名不兼容和优化组合冲突等结构性问题仍会阻断。正式交付应更新并重新生成 OptimizationConfig。

## 5. NUMA 或 CPU 亲和性失败

典型错误包括缺少 `numactl`、无法从 `hy-smi --showtopo` 获取 NUMA node、CPU list 格式错误，或 affinity 请求了当前进程不可用的 CPU。

检查命令：

```bash
numactl --hardware
hy-smi --showtopo
taskset -pc $$
```

处理原则：

- 自动 NUMA 要求 `hy-smi` 和 `numactl` 均在 `PATH` 中；
- `rank_affinity` 和 `rank_numa` 的键均为 `LOCAL_RANK`；
- 同时使用 NUMA 与 affinity 时，CPU 范围必须位于对应 NUMA node；
- `--disable-numa` 只关闭自动发现，不删除显式 `rank_numa`；
- 调度系统限制了 cpuset 时，应在允许范围内重新划分 affinity。

完整字段和覆盖规则见 [RuntimeConfig](runtime_config.md)。

## 6. Runner 不识别训练命令

`turbo-physai run` 只处理能够明确定位 Python 训练入口的命令。目前支持 Python、`torchrun` 和 `torchpack dist-run`。

以下形式不受支持：

- `python -c`；
- `torchrun --no-python`；
- 将 `bash -c`、管道、条件分支或其他 Shell 语法直接放在 `--` 后。

复杂作业编排应保留在 Shell 脚本或调度系统中，并在原 Python、`torchrun` 或 TorchPack 启动命令外层调用 `turbo-physai run`。

## 7. Replacement 在训练运行时报错

OptimizationReport 证明 Replacement 已安装，不代表所有运行时输入都满足优化实现的约束。根据 Python Traceback 定位 `turbo_physai/optimizations/` 或外部优化包中的函数，并使用触发问题的输入完成复现。

Catalog 声明 `runtime_condition` 时：

- 条件返回 `True`：执行优化实现；
- 条件返回 `False`：执行应用前的原实现；
- 条件返回非 `bool` 或条件函数抛出异常：错误向上传播；
- 优化实现抛出异常：错误向上传播，不自动切换到原实现。

段错误、进程 `abort` 和异步设备致命错误可能没有完整 Python Traceback，需要结合 core dump、HCU Runtime 日志和最小输入定位。

## 8. `torch.compile` 首步慢或持续重编译

- 首次执行可能包含图捕获和编译时间，不应计入稳态性能；
- 验证实际训练中的动态 shape、布尔分支、`None`/非 `None` 和训练/评测路径；
- 使用 `TORCH_LOGS=recompiles` 检查预热后是否持续生成新图；
- 性能对比应使用相同编译缓存、输入序列、预热次数和统计窗口。

## 9. DataLoader 首批数据等待时间过长

采用 `spawn` 时，worker 会重新导入模型插件和优化模块，初始化时间可能显著增加。应优先使用模型 README 中经过验证的启动方式。

BEVFormer 的训练 Group 默认将 DataLoader multiprocessing start method 设置为 `fork`。需要覆盖时可设置：

```bash
turbo-physai run \
  --model bevformer \
  --set TURBO_PHYSAI_DATALOADER_START_METHOD=spawn \
  -- \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

修改后必须重新验证数据加载正确性、首批耗时和稳定性能。

## 10. 非 Rank 0 没有报告文件

这是预期行为。每个 rank 都会完成自身的检查和应用，但仅 Rank 0 写入 JSON 和 Markdown 报告。当前报告不聚合其他 rank 的运行结果，应结合各 rank 的 `TURBO_PHYSAI_OPTIMIZATION_COMPLETED` 输出确认启动状态。

## 11. 回滚失败

回滚失败表示框架无法确认 Group 已恢复到快照状态。此时进程中的 Python 对象状态不可继续信任，框架会停止后续 Group、写入报告并抛出 `OptimizationRollbackError`。

处理步骤：

1. 保存 OptimizationReport 和完整 Traceback；
2. 终止当前训练进程；
3. 定位失败的 Replacement 及 restore 结果；
4. 修复后使用全新进程重新启动。

不要在发生回滚失败的进程中继续训练或再次调用 `apply()`。

## 12. 高级 Python API 问题

### 重复调用 `apply()`

典型错误：

```text
turbo_physai.apply() may only be called once per process
```

`apply()` 每个进程只能调用一次，失败后也不支持在原进程重试。默认使用 `turbo-physai run` 时，Runner 会在每个训练 rank 中管理调用时机。直接集成 Python API 时，应保证调用早于模型业务导入，并避免模块顶层代码被 worker 重复执行。

### `check()` 产生初始化开销

`turbo_physai.check()` 不安装 Replacement，但会导入 target 和 Replacement 模块；`wrap()` 还会执行 Wrapper 构造函数。导入动态库、注册算子和 Wrapper 构造产生的外部副作用不一定可以撤销，因此优化模块的导入和 Wrapper 构造应避免不可逆的全局修改。
