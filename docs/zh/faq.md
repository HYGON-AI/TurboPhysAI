# 常见问题

## 产品镜像是否包含模型源码、数据集和权重？

不包含。产品镜像提供 TurboPhysAI、HCU 软件环境和已验证的模型依赖。模型源码、数据集、预训练权重和训练输出目录需要按照[模型支持清单](models/support_list.md)中的对应模型说明准备并挂载到容器中。

## 如何通过原训练命令启用优化？

已支持模型使用：

```bash
turbo-physai run \
  --model <model-name> \
  -- \
  <original-training-command>
```

Runner 自动选择随包交付的 OptimizationConfig 和 RuntimeConfig，并在每个 Python 训练进程中应用优化。原训练代码无需增加 TurboPhysAI 初始化语句。

RuntimeConfig 可以设置或清除环境变量，并控制 NUMA 绑定。复杂 Shell、资源限制和作业调度逻辑仍由模型启动脚本或作业系统负责。

## 不指定 `--model` 会加载什么配置？

Runner 加载内置 common OptimizationConfig，其中只包含与具体模型无关的公共优化。该方式不自动加载 RuntimeConfig；训练环境和启动参数由原命令、外部脚本或显式传入的 `--runtime-config` 提供。

## TurboPhysAI 会修改模型源码吗？

`turbo-physai run` 和 `turbo_physai.apply()` 不写入模型仓库文件。优化应用发生在当前 Python 进程中：框架将声明的 `target` 和 `aliases` 指向 Replacement，或在原对象外层安装 Wrapper。

模型源码、依赖和数据准备仍应满足对应模型应用说明中的前置条件。

## 如何确认优化是否生效？

查看本次运行生成的 OptimizationReport：

- Preparation 中决策为 `apply` 的 Group 已进入执行阶段；
- Execution 中状态为 `applied` 的 Group 已完成安装；
- `block`、`rolled_back`、`failed` 或 `not_started` 表示对应优化未完整生效。

报告结构和状态定义见[优化应用报告](user_guide/report.md)。

## Group 被阻断会停止其他优化吗？

通常不会。被阻断的 Group 及其下游依赖不会执行，其他独立 Group 继续处理；Group 应用失败但成功回滚时，其他独立 Group 也可以继续。

回滚失败表示进程状态无法可靠恢复，此时框架会停止后续执行。具体排查方法见[问题排查：Group 未应用](user_guide/troubleshooting.md#2-group-未应用)，处理规则见[优化检查、执行与回滚](design/execution.md)。

## 模型 commit 变化但 `target` Hash 相同，可以应用吗？

commit 不一致本身只会使 OptimizationReport 中的 `project.commit` 检查产生 `warning`。如果 `target` Hash、签名、对象身份和 Group 兼容条件等其余检查全部通过，对应 Group 仍可应用。

正式支持新的模型基线前，应完成正确性和性能验证，并重新生成 OptimizationConfig。

## `apply()` 返回后优化是否已经安装？

是。`apply()` 返回时，Execution 状态为 `applied` 的 `target` 和 `aliases` 已经指向优化对象，不需要等待模型第一次调用。

模型第一次调用该入口时，才会执行 Replacement 中的具体计算。`apply()` 是高级 Python API；多进程训练默认使用 `turbo-physai run`。

## 可以在一个进程中切换 OptimizationConfig 吗？

不可以。每个进程只允许调用一次 `apply()`。更换 OptimizationConfig 或在失败后重新应用，必须启动新的 Python 进程。

## TurboPhysAI 能与第三方运行时替换组件共存吗？

可以，但需要固定组件版本和加载顺序。同一目标被多个组件修改时，TurboPhysAI 可能因对象证据不一致而阻断对应 Group，也可能在自身应用后被其他组件再次覆盖。

正式交付前应基于最终组合重新生成 OptimizationConfig，并验证数值和性能。检查失败的处理方法见[问题排查](user_guide/troubleshooting.md#3-targetalias-或-hash-检查失败)，加载顺序和覆盖规则见[兼容性管理](design/compatibility.md)。

## 能接入自定义原生算子吗？

可以。仓库内原生算子源码通过统一构建链路编译进 `turbo_physai.ops`。接入流程、PyBind 注册方式和验证要求见[自定义算子接入](developer_guide/custom_operator.md)。

外部动态库由其发布包或产品镜像负责安装和加载；TurboPhysAI 不管理外部动态库的安装、ABI、卸载和运行时回滚。
