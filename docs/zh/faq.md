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

普通运行时 Group 的决策为 `block` 时，该 Group 不执行；依赖它的下游 Group 记为 `not_started`，其他独立 Group 继续执行。Group 应用失败但成功回滚时，其他独立 Group 也可以继续。

导入兼容 Group 用于建立模型模块的可导入条件。该类 Group 失败时，后续运行时优化无法安全解析目标，因此会被阻断。回滚失败表示进程状态无法可靠恢复，框架将停止后续执行并抛出异常。

## 模型 commit 变化但 `target` Hash 相同，可以应用吗？

commit 不一致本身只会使 OptimizationReport 中的 `project.commit` 检查产生 `warning`。如果 `target` Hash、签名、对象身份和 Group 兼容条件等其余检查全部通过，对应 Group 仍可应用。

正式支持新的模型基线前，应完成正确性和性能验证，并重新生成 OptimizationConfig。

## `apply()` 返回后优化是否已经安装？

是。`apply()` 返回时，Execution 状态为 `applied` 的 `target` 和 `aliases` 已经指向优化对象，不需要等待模型第一次调用。

模型第一次调用该入口时，才会执行 Replacement 中的具体计算。`apply()` 是高级 Python API；多进程训练默认使用 `turbo-physai run`。

## 可以在一个进程中切换 OptimizationConfig 吗？

不可以。每个进程只允许调用一次 `apply()`。更换 OptimizationConfig 或在失败后重新应用，必须启动新的 Python 进程。

## TurboPhysAI 能与第三方运行时替换组件共存吗？

TurboPhysAI 会检查应用时解析到的实际对象，但不能控制未知第三方组件的加载顺序和内部行为。

- 第三方组件先修改同一 `target`：对象证据不一致时，TurboPhysAI 默认阻断对应 Group。开发人员确认变更安全后，只能对框架标记为可放行的检查临时使用 `force_groups`；
- 第三方组件在 TurboPhysAI 之后修改同一 `target`：后一次赋值会覆盖 TurboPhysAI 的修改，`force_groups` 不能改变该加载顺序；
- 正式交付：应固定组件版本和初始化顺序，并基于最终组合重新生成 OptimizationConfig、验证数值和性能。

详细规则见[兼容性管理](design/compatibility.md)。

## 能接入自定义原生算子吗？

可以。仓库内原生算子源码通过统一构建链路编译进 `turbo_physai.ops`。接入流程、PyBind 注册方式和验证要求见[自定义算子接入](developer_guide/custom_operator.md)。

外部动态库由其发布包或产品镜像负责安装和加载；TurboPhysAI 不管理外部动态库的安装、ABI、卸载和运行时回滚。
