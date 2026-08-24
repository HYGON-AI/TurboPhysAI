# CLI 参考

产品镜像和源码安装环境均使用 `turbo-physai`。仅在尚未安装组件的源码开发目录中，可使用 `PYTHONPATH=. python -m turbo_physai.cli` 调用同一入口。

## optimization init

创建空白的外部优化开发工程：

```bash
turbo-physai optimization init <name> [--output <directory>]
```

- `name`：模型或优化项目名称，必须以字母开头，只包含字母、数字、`-`、`_`；
- `--output`：输出目录；未指定时，框架将名称转为小写、将 `-` 转为 `_`，并使用 `<normalized_name>_optimization`；
- 已存在路径不会覆盖。

## optimization validate

```bash
turbo-physai optimization validate <optimization-config.yaml>
```

加载 OptimizationConfig，校验 Schema 和字段类型，展开 `extends`，并导入 `optimization_modules`。成功时输出 OptimizationConfig ID 和版本。

该命令不检查模型仓库和 target Hash。声明模块及其 Python 依赖必须能够在当前环境中导入。

## optimization show

```bash
turbo-physai optimization show <optimization-config.yaml>
```

输出加载并展开后的 JSON 表示。该命令与 `optimization validate` 使用相同的加载过程，也会导入 `optimization_modules`。

## optimization check

```bash
turbo-physai optimization check <generated-optimization-config.yaml> [--repo <model-repo>]
```

检查已生成的 OptimizationConfig 与当前模型仓库是否一致。`--repo` 默认为当前目录。该命令要求模型工作区干净，并检查 Group 依赖闭包、执行顺序、Group 组合以及 target 的 source/AST Hash。模型仓库 commit 不同本身不会导致检查失败。

## optimization diff

```bash
turbo-physai optimization diff <left.yaml> <right.yaml>
```

加载并展开两个 OptimizationConfig，比较其 JSON 表示并输出 unified diff。该命令比较的是解析后的配置，不是 YAML 原始文本。

## optimization generate

```bash
turbo-physai optimization generate \
  --recipe <recipe.yaml> \
  --repo <model-repo> \
  --commit <validated-commit> \
  --output <generated-optimization-config.yaml> \
  [--force]
```

- 模型仓库 HEAD 必须等于 `--commit`；
- 模型仓库必须干净；
- `--commit` 用于本次生成的基线确认，并作为非阻断的参考信息写入最终 YAML；
- 展开 `extends` 和 Group 依赖，确定最终 Group 顺序；
- 检查 Group 组合冲突和模型 Replacement 对公共 Replacement 的直接引用；
- 从当前模型仓库提取 target source/AST Hash；
- 输出已存在时默认返回错误；
- `--force` 明确覆盖输出文件。

已生成的 OptimizationConfig 使用 `optimization check` 重新核对。

## run

准备 RuntimeConfig，并在每个 Python 训练进程中应用 OptimizationConfig 后执行原训练入口：

```bash
turbo-physai run \
  [--model <model-name>] \
  [--optimization-config <model-optimization-config.yaml>] \
  [--runtime-config <runtime.yaml>] \
  [--report-dir <directory>] \
  [--force-group GROUP_ID] \
  [--disable-group GROUP_ID] \
  [--set NAME=VALUE] \
  [--disable-numa] \
  -- <training-command>
```

| 参数 | 含义 |
| --- | --- |
| `--model` | 内置模型名称；自动选择该模型的 OptimizationConfig 和 RuntimeConfig |
| `--optimization-config` | 显式 OptimizationConfig 路径；优先于 `--model` 选择结果 |
| `--runtime-config` | 显式 RuntimeConfig 路径；优先于 `--model` 选择结果 |
| `--report-dir` | 报告目录，默认 `turbophysai_reports` |
| `--force-group GROUP_ID` | 对指定且已启用 Group 的可放行检查进行一次性覆盖，可重复 |
| `--disable-group GROUP_ID` | 本次启动不应用指定 Group；可一次指定多个 Group，也可重复使用 |
| `--set NAME=VALUE` | 覆盖或增加非空环境变量，可重复 |
| `--disable-numa` | 关闭本次启动的 NUMA 绑定；默认开启 |

命令在 `--` 之后逐字透传，不被解析或改写，因此对启动器没有限制：`torchrun`、
`torchpack dist-run`、DeepSpeed、accelerate、`mpirun`、`srun` 以及自有的 Shell 启动
脚本都可直接使用。每个训练 rank 的解释器在启动时自动激活优化。

例外是 `python -E`、`-I`、`-S`：这三个标志会使启动钩子不被加载，命令中出现它们时
`turbo-physai run` 直接报错，而不是让训练以未优化状态运行。

Shell 语法本身仍不由 `turbo-physai run` 解析。`source`、`ulimit`、管道和条件分支应
保留在作业脚本中，由该脚本调用 `turbo-physai run`。完整字段和覆盖规则见
[RuntimeConfig 使用指南](../user_guide/runtime_config.md)。

配置选择顺序如下：

1. 显式 `--optimization-config` 或 `--runtime-config`；
2. `--model` 对应目录下的 `configs/optimization.yaml` 和可选的 `configs/runtime.yaml`；
3. 未指定模型和 OptimizationConfig 时，使用内置 `optimizations/common/configs/optimization.yaml`，不加载 RuntimeConfig。

模型名称不区分大小写，`-` 会按 `_` 处理。例如 `--model BEVFormer` 与 `--model bevformer` 等价。

```bash
turbo-physai run \
  --model bevformer \
  -- \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

TorchPack 示例：

```bash
turbo-physai run \
  --model bevfusion \
  -- \
  torchpack dist-run -np 8 python tools/train.py path/to/config.py
```

`--force-group` 仅对本次启动有效，且 Group 必须已被当前 OptimizationConfig 启用。该参数只能覆盖框架明确标记为可放行的证据类检查。目标不存在、Alias 身份冲突、签名不兼容和优化冲突等结构性问题仍会阻断。

`--disable-group` 仅对本次启动有效，不修改 OptimizationConfig。依赖被禁用 Group 的其他 Group 同时跳过。报告分别使用 `disabled_by_user` 和 `dependency_disabled` 记录两类原因。同一 Group 不能同时强制放行和禁用。

`run` 用 `exec` 替换自身执行训练命令，进程树中不留中间进程。`SIGINT`、`SIGTERM` 等信号由训练进程直接接收，不经过转发。

## 返回码

| 返回码 | 含义 |
| --- | --- |
| `0` | 命令成功，或训练进程正常退出 |
| `2` | 训练进程执行前发生参数、OptimizationConfig、RuntimeConfig、生成检查或覆盖策略错误 |
| `91` | 某个 rank 无法应用 OptimizationConfig，主动终止而不是以未优化状态继续训练 |
| 训练进程返回码 | `run` 执行训练后，退出码即训练进程自身的退出码，包括信号导致的 `128 + signal` |

非 `run` 命令的具体错误原因写入标准错误输出。`run` 执行训练后，训练日志和异常由训练进程直接输出。
