# 优化应用报告

OptimizationReport 用于确认一次训练启动实际加载了哪些配置、各项优化是否完成安装，以及未完成安装的原因。报告将应用前检查、Group 决策、执行结果和回滚结果集中记录，便于训练验收和问题定位。

状态为 `applied` 时，该 Group 声明的 `target` 和 `aliases` 已指向优化对象，后续模型执行链路通过这些入口发起的调用将使用优化后的对象。模型数值、精度和性能仍需按照对应模型说明完成验证。

## 1. 输出文件

`turbo-physai run` 默认将报告写入 `turbophysai_reports`。可通过 `--report-dir` 修改目录：

```bash
turbo-physai run \
  --model bevformer \
  --report-dir ./turbophysai_reports \
  -- \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

一次 `turbo-physai run` 生成一份 OptimizationReport，并以相同 Run ID 输出 JSON 和
Markdown 两种格式：

```text
turbophysai_reports/
├── optimization_report-<run-id>.json
└── optimization_report-<run-id>.md
```

- JSON 保存完整结构化数据，适用于自动检查、归档和进一步分析；
- Markdown 展示主要配置、异常检查、Group 决策和执行结果，适用于人工查看。

## 2. 报告结构

Markdown 报告包含以下内容：

| 部分 | 内容 |
| --- | --- |
| 基本信息 | Run ID、OptimizationConfig ID 和版本、实际配置路径、RuntimeConfig 路径 |
| `Summary` | 各类决策和执行状态的数量 |
| `Configuration Checks` | OptimizationConfig 级检查，例如模型仓库 commit |
| `Preparation` | 每个 Group 的依赖、应用决策、原因和异常检查 |
| `Execution` | 已进入执行阶段的 Group 状态和错误信息 |

JSON 在此基础上保存运行环境快照、全部 Group 检查、Replacement 成员结果、目标修改记录、回滚结果和报告文件路径。需要进行自动化判断时，应以 JSON 为准。

OptimizationConfig 和 RuntimeConfig 均以本次实际加载的绝对路径记录。未使用 RuntimeConfig 时，Markdown 显示 `not used`，JSON 中的 `runtime_config_path` 为 `null`。

## 3. 检查状态

检查结果使用以下状态：

| 状态 | 含义 |
| --- | --- |
| `pass` | 检查通过 |
| `warning` | 存在差异，但该差异不阻断应用 |
| `fail` | 检查未通过 |
| `unknown` | 当前环境无法取得检查所需信息 |
| `not_applicable` | 该检查不适用于当前对象 |

OptimizationConfig 级检查在 `Configuration Checks` 中记录一次。例如，模型 commit 与配置记录不一致时，`project.commit` 为 `warning`，Group 是否可以应用仍由 target 证据和其他检查决定。

Markdown 在 `Preparation` 中展开 `warning`、`fail` 和 `unknown` 的 Group 检查。JSON 保存全部检查，包括 target、Replacement、aliases、签名、Hash 和 Group 兼容条件。

## 4. 应用决策

`Preparation` 为每个配置项记录一种决策：

| 决策 | 含义 |
| --- | --- |
| `apply` | 检查通过，Group 进入执行列表 |
| `skip` | Group 未启用，不进入执行列表 |
| `block` | Group 存在未满足的应用条件，不进入执行列表 |

`block` 只隔离该 Group 及依赖它的 Group。其他无依赖关系且检查通过的 Group 仍可执行。

## 5. 执行状态

`Execution` 只记录进入执行阶段的 Group：

| 状态 | 含义 |
| --- | --- |
| `applied` | Group 的全部 Replacement 已完成安装，声明的入口已指向优化对象 |
| `rolled_back` | 应用过程中发生异常，Group 已恢复到应用前状态 |
| `failed` | Group 快照失败，或应用异常后未能完整恢复 |
| `not_started` | 依赖 Group 未成功应用，或此前发生终止性执行错误 |

`skip` 和 `block` 是应用决策，不是执行状态，因此不会作为 Group 状态出现在 `Execution` 中。

## 6. Summary 字段

| 字段 | 统计范围 |
| --- | --- |
| `applied` | 执行状态为 `applied` 的 Group |
| `skipped` | 决策为 `skip` 的 Group |
| `blocked` | 决策为 `block` 的 Group |
| `failed` | 执行状态为 `failed` 的 Group |
| `rolled_back` | 执行状态为 `rolled_back` 的 Group |
| `not_started` | 执行状态为 `not_started` 的 Group |

一次完整成功的应用应满足：

```text
blocked = 0
failed = 0
rolled_back = 0
not_started = 0
```

`skipped` 可以大于 0，表示部分 Group 不参与本次执行。常见原因包括配置中的
`enabled: false`、通过 `--disable-group` 临时禁用，以及所依赖的 Group 被禁用。具体原因见
`Planning` 中对应 Group 的 `Reason`；预期执行的 Group 应全部计入 `applied`。

## 7. 异常和回滚

Group 应用失败后，框架按快照恢复该 Group 的全部成员：

- 回滚成功：Group 状态为 `rolled_back`，其他独立 Group 可以继续执行；
- 回滚失败：Group 状态为 `failed`，后续 Group 标记为 `not_started`，框架写入报告后抛出 `OptimizationRollbackError`；
- 报告文件写入失败：抛出 `ReportWriteError`。

Group 决策为 `block` 或状态为 `rolled_back` 时，`apply()` 可以正常返回报告。因此，不能仅根据训练进程是否启动判断所有优化均已完成安装。

高级 Python 集成可以直接检查 Summary：

```python
import turbo_physai

report = turbo_physai.apply(
    optimization_config_path="./configs/optimization.yaml",
)
summary = report.summary

if any(summary[name] for name in (
    "blocked",
    "failed",
    "rolled_back",
    "not_started",
)):
    raise RuntimeError("not all enabled Optimization Groups were applied")
```

## 8. 临时放行

开发人员确认某项可放行证据变化安全后，可以对本次启动指定 Group：

```bash
turbo-physai run \
  --model bevformer \
  --force-group <group-id> \
  -- \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

`--force-group` 仅覆盖框架标记为可放行的检查，例如部分工作区状态、依赖版本或 target Hash 证据问题。目标不存在、alias 身份冲突、签名不兼容和优化组合冲突等结构性问题不能放行。

临时放行只对本次启动有效。正式交付应在经过验证的模型基线上重新生成 OptimizationConfig。
