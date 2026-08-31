# OptimizationConfig 使用指南

OptimizationConfig 用于选择一次训练需要启用的 Optimization Group，并保存目标代码的兼容性证据。`turbo-physai run` 在训练进程启动时加载该配置、执行应用前检查，并安装检查通过的优化。

OptimizationConfig 不包含环境变量或 NUMA 设置。这些启动参数由 [RuntimeConfig](runtime_config.md) 管理。

## 1. 选择配置

### 1.1 使用内置模型配置

指定已支持的模型后，Runner 自动加载该模型随包交付的 OptimizationConfig：

```bash
turbo-physai run \
  --model bevformer \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

模型名称不区分大小写，名称中的 `-` 按 `_` 处理。可用模型及训练命令见[模型支持清单](../models/support_list.md)。

### 1.2 使用公共优化配置

未指定 `--model` 和 `--optimization-config` 时，Runner 加载随包交付的公共 OptimizationConfig：

```bash
turbo-physai run \
  python tools/train.py path/to/config.py
```

该方式只启用公共优化，不加载模型专用优化和 RuntimeConfig。

### 1.3 使用显式配置

外部优化包或自定义交付可通过路径指定 OptimizationConfig：

```bash
turbo-physai run \
  --optimization-config ./configs/optimization.yaml \
  python tools/train.py path/to/config.py
```

`--optimization-config` 优先于 `--model` 对应的 OptimizationConfig。若同时指定 `--model` 且未指定 `--runtime-config`，Runner 仍会加载该模型的 RuntimeConfig。

## 2. 配置结构

以下内容节选自随包交付的 BEVFormer OptimizationConfig。配置生成工具写入的 `trust` 内容未在示例中展开：

```yaml
schema_version: turbophysai/optimization-config/v1
kind: OptimizationConfig

metadata:
  id: model.bevformer.base.hcu
  version: "1.0.0"
  description: Complete validated HCU optimization recipe for official BEVFormer

model:
  name: BEVFormer

optimization_modules:
  - turbo_physai.optimizations.models.bevformer.catalog

compatibility:
  commits:
    - 66b65f3a1f58caf0507cb2a971b9c0e7f842376c

optimization_groups:
  - id: bevformer.mdc
    enabled: true
```

### 2.1 基本信息

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 配置格式版本，当前为 `turbophysai/optimization-config/v1` |
| `kind` | 配置类型，固定为 `OptimizationConfig` |
| `metadata.id` | OptimizationConfig 的唯一 ID |
| `metadata.version` | OptimizationConfig 的交付版本 |
| `metadata.description` | 配置用途说明 |
| `model.name` | 配置对应的模型名称；公共配置可以不填写 |

### 2.2 优化选择

| 字段 | 含义 |
| --- | --- |
| `optimization_modules` | 提供 Optimization Group 声明的 Python 模块路径 |
| `extends` | 引用的基础 OptimizationConfig ID |
| `optimization_groups` | 本配置选择的 Optimization Group |
| `optimization_groups[].id` | Group ID，必须与 Catalog 中的声明一致 |
| `optimization_groups[].enabled` | 是否启用该 Group，默认值为 `true` |
| `optimization_groups[].options` | 传递给明确读取配置的 Wrapper；普通 `replace` 不读取该字段 |

Group 的 target、aliases、Replacement 和 `depends_on` 由 Catalog 定义，不在 OptimizationConfig 中重复保存。加载配置时，框架会导入 `optimization_modules`，并根据 Group ID 查找对应声明。

### 2.3 兼容性证据

| 字段 | 含义 |
| --- | --- |
| `compatibility.commits` | 模型配置生成时使用的模型 commit，运行时用于记录基线差异；公共配置不填写 |
| `compatibility.allow_dirty` | 是否允许模型工作区存在未提交修改 |
| `compatibility.dependencies` | 需要检查的 Python 包版本 |
| `compatibility.backend` | 允许的运行后端 |
| `compatibility.repository` | 允许的模型仓库标识 |
| `optimization_groups[].trust` | target 的源码、AST 或原生扩展文件指纹 |

模型 commit 不匹配时，报告中的 `project.commit` 状态为 `warning`，不会单独阻断 Group。框架仍会根据 target 证据和其他显式兼容条件判断该 Group 是否可以应用。

Python target 使用 source Hash 和 AST Hash。两者任意一个匹配即可通过身份检查。没有 Python 源码的原生扩展入口使用扩展文件指纹。aliases 通过与主 target 的对象身份关系检查，不单独保存 Hash。

## 3. 继承和依赖

`extends` 用于复用基础 OptimizationConfig。例如模型配置可以引用 `common.hcu.base`，再增加模型专用 Group。

Group 之间只有在 Catalog 明确声明 `depends_on` 时才形成依赖。配置生成工具会展开 `extends`，补齐依赖 Group，并按依赖关系确定稳定顺序。运行时若依赖 Group 未成功应用，下游 Group 的执行状态为 `not_started`；其他无依赖关系的 Group 不受影响。

生成后的 OptimizationConfig 已包含最终选择的 Group，但仍依赖对应版本的 Catalog。配置与提供 Group 声明的 TurboPhysAI 或外部优化包应作为同一次交付进行版本管理。

## 4. 校验配置

校验 YAML Schema、字段类型、继承关系和声明模块导入：

```bash
turbo-physai optimization validate ./configs/optimization.yaml
```

核对已生成配置与模型仓库中的 target 证据：

```bash
turbo-physai optimization check \
  ./configs/optimization.yaml \
  --repo /path/to/model-repository
```

`optimization check` 要求模型工作区干净，并检查 Group 依赖闭包、顺序、组合冲突及 target 证据。两条命令的完整参数见 [CLI 参考](../reference/cli.md)。

## 5. 配置维护

- 调整 Group 选择时，修改开发侧 recipe 并重新生成 OptimizationConfig；
- 升级模型基线时，在经过验证的新 commit 上重新生成配置；
- 升级 Replacement 时，同步升级 TurboPhysAI 或外部优化包并重新完成模型验证；
- 不直接修改生成配置中的 `trust`；
- 不将 `--force-group` 作为正式交付配置。

OptimizationConfig 的创建和生成流程见[优化配置生成](../developer_guide/optimization_config_generation.md)。

## 6. 高级 Python 接口

需要自行管理 Python 训练入口时，可以直接调用 `turbo_physai.apply()`：

```python
import turbo_physai

report = turbo_physai.apply(
    model="bevformer",
    disable_groups=["bevformer.compile.encoder", "bevformer.grid_mask"],
    log_report=True,
)
```

每个训练进程只能调用一次 `apply()`，且应在导入目标模型前执行。多进程训练推荐使用 `turbo-physai run`，由 Runner 在每个训练 rank 中完成调用。`log_report` 默认值为 `False`；设为 `True` 时，Rank 0 将 OptimizationReport 输出到标准日志。该参数不影响报告对象的返回值，也不改变优化检查和应用行为。

完整参数、配置选择顺序、返回值、调用约束和 `turbo_physai.check()` 说明见[Python API 参考](../reference/python_api.md)。
