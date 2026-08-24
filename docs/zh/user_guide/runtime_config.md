# RuntimeConfig

RuntimeConfig 描述训练命令所需的环境变量、CPU 亲和性和 NUMA 绑定。它只负责运行环境，不选择代码优化；代码优化由 OptimizationConfig 管理。

使用 `--model` 启动内置模型时，TurboPhysAI 自动加载该模型随包交付的 RuntimeConfig。部署环境需要调整网卡、设备范围或进程绑定时，可通过命令行覆盖；需要交付完整自定义配置时，可显式指定 RuntimeConfig 文件。

## 1. 配置示例

```yaml
schema_version: turbophysai/runtime-config/v1
kind: RuntimeConfig

environment:
  unset:
    - NCCL_TOPO_FILE
    - NCCL_RINGS
  set:
    HIP_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    ENABLE_TORCH_PROFILER: "0"

process:
  numa: auto
  rank_affinity:
    "0": "0-15"
  rank_numa: {}
```

字段含义如下：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | RuntimeConfig 格式版本，当前为 `turbophysai/runtime-config/v1` |
| `kind` | 配置类型，固定为 `RuntimeConfig` |
| `environment.unset` | 从启动环境中删除指定变量 |
| `environment.set` | 设置环境变量；非字符串值在加载时转换为字符串 |
| `process.numa` | 是否根据 HCU 拓扑自动选择 NUMA node；支持 `false`、`true` 和 `auto` |
| `process.rank_affinity` | 按 local rank 设置 CPU 亲和性，值采用 Linux CPU list 格式 |
| `process.rank_numa` | 按 local rank 显式设置 CPU 与内存所在的 NUMA node |

`process.numa: true` 与 `process.numa: auto` 当前行为一致。使用 `auto` 更能直接表达“根据设备拓扑自动选择”的含义。

## 2. 环境变量

Runner 按以下顺序构造训练命令环境：

1. 继承当前 Shell 环境；
2. 删除 `environment.unset` 中的变量；
3. 写入 `environment.set`；
4. 写入命令行 `--set` 指定的变量。

同名变量以后写入的值为准。例如：

```bash
turbo-physai run \
  --model bevformer \
  --set HIP_VISIBLE_DEVICES=0,1,2,3 \
  --set NCCL_SOCKET_IFNAME=ens19f0 \
  -- \
  torchrun --nproc-per-node=4 tools/train.py path/to/config.py
```

上例中的两个 `--set` 会覆盖当前 Shell 和模型 RuntimeConfig 中的同名变量。网络接口、设备范围和 rendezvous 地址等依赖部署环境的参数，建议在作业脚本或调度系统中显式传入。

## 3. CPU 亲和性

`rank_affinity` 的键是 `LOCAL_RANK`，值采用 Linux CPU list 格式，例如 `0-15`、`0-7,96-103`：

```yaml
process:
  rank_affinity:
    "0": "0-15"
    "1": "16-31"
```

也可以在启动时覆盖单个 rank：

```bash
turbo-physai run \
  --model bevformer \
  --set-rank-affinity 0=0-15 \
  --set-rank-affinity 1=16-31 \
  -- \
  torchrun --nproc-per-node=2 tools/train.py path/to/config.py
```

Runner 在每个训练 rank 内调用 Linux CPU affinity 接口。命令行设置覆盖 RuntimeConfig 中相同 rank 的值。CPU list 格式错误，或请求的 CPU 不在当前进程允许范围内时，Runner 会终止该 rank 并报告错误。

同时使用 NUMA 与 CPU affinity 时，CPU affinity 必须位于对应 NUMA node 为该进程提供的 CPU 范围内。

## 4. NUMA 绑定

### 4.1 自动绑定

```yaml
process:
  numa: auto
```

每个 rank 根据 `LOCAL_RANK` 找到 `HIP_VISIBLE_DEVICES` 中对应的 HCU，再从 `hy-smi --showtopo` 读取该设备所属的 NUMA node。Runner 随后通过以下等价方式重新进入当前 rank：

```text
numactl --cpunodebind=<node> --membind=<node> python -m turbo_physai.runner ...
```

自动绑定要求运行环境同时提供 `hy-smi` 和 `numactl`，并且 `hy-smi --showtopo` 能返回 HCU 的 NUMA node。

### 4.2 显式绑定

当机器拓扑固定或自动发现不适用时，可以按 local rank 指定 NUMA node：

```yaml
process:
  numa: false
  rank_numa:
    "0": 0
    "1": 0
    "2": 1
    "3": 1
```

命令行覆盖方式如下：

```bash
turbo-physai run \
  --model bevformer \
  --set-rank-numa 0=0 \
  --set-rank-numa 1=0 \
  -- \
  torchrun --nproc-per-node=2 tools/train.py path/to/config.py
```

显式 `rank_numa` 的优先级高于自动发现：配置了显式 node 的 rank 不再执行拓扑推导。

`--enable-numa` 和 `--disable-numa` 只覆盖 `process.numa` 的自动绑定开关。`--disable-numa` 不会删除 RuntimeConfig 或 `--set-rank-numa` 中的显式绑定；如需完全禁用 NUMA，应同时确保未配置 `rank_numa`。

## 5. 配置选择与覆盖

```bash
turbo-physai run \
  --model bevformer \
  --runtime-config ./configs/runtime.yaml \
  --set NAME=VALUE \
  --disable-numa \
  -- \
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

选择规则如下：

1. 指定 `--runtime-config` 时加载该文件；
2. 否则，指定 `--model` 时自动查找模型随包配置；
3. 两者都未指定时，不加载 RuntimeConfig；
4. 文件加载后，`--set`、`--set-rank-affinity`、`--set-rank-numa` 和 NUMA 开关覆盖对应配置项。

未加载 RuntimeConfig 不影响 common OptimizationConfig 的选择。两类配置由 Runner 独立解析。

## 6. 支持的训练命令

`turbo-physai run` 支持以下启动形式：

```text
python script.py [参数]
python -I script.py [参数]
python -m package.module [参数]
torchrun [torchrun 参数] script.py [参数]
torchrun [torchrun 参数] -m package.module [参数]
torchpack dist-run [TorchPack 参数] python [Python 参数] script.py [参数]
torchpack dist-run [TorchPack 参数] python [Python 参数] -m package.module [参数]
```

Runner 保留原启动器参数，只将训练入口改写为 `turbo_physai.runner`。每个训练 rank 先准备 NUMA 和 CPU affinity、应用 OptimizationConfig，再通过 `runpy` 在同一 Python 进程执行原训练脚本或模块。

当前支持的 Python 选项包括 `-I`、`-u`、`-B`、`-O/-OO`、`-X`、`-W` 和 `-m`。不支持 `python -c`、`torchrun --no-python` 或无法确定 Python 训练入口的命令。

Shell 语法不由 Runner 解析。`source`、`ulimit`、管道、条件分支和其他系统编排应继续保留在作业脚本中，并由该脚本调用 `turbo-physai run`。

## 7. 失败与停止

- RuntimeConfig 格式错误、CPU list 无效、NUMA node 无效或缺少必要系统命令时，训练在入口执行前失败；
- 首次 `Ctrl+C` 会将 `SIGINT` 转发给训练进程组，并等待最多 30 秒；
- 训练未退出时，Runner 自动发送 `SIGTERM`；继续等待 5 秒仍未退出时发送 `SIGKILL`；
- 等待期间再次按 `Ctrl+C` 会提前进入下一停止阶段，正常停止不要求重复操作。

RuntimeConfig 的修改可能影响通信、CPU 调度和算子选择。交付前应在目标机器上重新验证训练正确性和稳定性能。
