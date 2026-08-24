# RuntimeConfig

RuntimeConfig 描述训练命令所需的环境变量和进程设置。它只负责运行环境，不选择代码优化；代码优化由 OptimizationConfig 管理。

使用 `--model` 启动内置模型时，TurboPhysAI 自动加载该模型随包交付的 RuntimeConfig。部署环境需要调整网卡或设备范围时，可通过 `--set` 覆盖；需要交付完整自定义配置时，可显式指定 RuntimeConfig 文件。

## 1. 配置示例

```yaml
schema_version: turbophysai/runtime-config/v1
kind: RuntimeConfig

environment:
  unset: []
  set:
    HIP_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    ENABLE_TORCH_PROFILER: "0"

process:
  numa: true
```

字段含义如下：

| 字段 | 说明 |
| --- | --- |
| `schema_version` | RuntimeConfig 格式版本，当前为 `turbophysai/runtime-config/v1` |
| `kind` | 配置类型，固定为 `RuntimeConfig` |
| `environment.unset` | 从启动环境中删除指定变量 |
| `environment.set` | 设置环境变量；非字符串值在加载时转换为字符串 |
| `process.numa` | 是否启用 NUMA 绑定；默认为 `true` |

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

## 3. NUMA 绑定

NUMA 只有开启和关闭两种使用方式：

- 默认开启；
- 使用 `--disable-numa` 关闭本次启动的 NUMA 绑定。

## 4. 配置选择与覆盖

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
4. 文件加载后，`--set` 覆盖环境变量，`--disable-numa` 关闭本次启动的 NUMA 绑定。

未加载 RuntimeConfig 不影响 common OptimizationConfig 的选择。两类配置由 Runner 独立解析。

## 5. 支持的训练命令

`turbo-physai run` 的命令在 `--` 之后逐字透传给 `exec`，不被解析或改写，因此启动形式不受限制：

```text
python script.py [参数]
python -m package.module [参数]
torchrun [torchrun 参数] script.py [参数]
torchpack dist-run [TorchPack 参数] python script.py [参数]
deepspeed [DeepSpeed 参数] script.py [参数]
accelerate launch [参数] script.py [参数]
srun / mpirun [参数] python script.py [参数]
bash train.sh
```

每个训练 rank 的解释器在启动时由标准库 `site` 自动导入 TurboPhysAI 的启动钩子，完成运行设置并应用 OptimizationConfig，再执行原训练入口。启动器进程本身不应用优化。

例外是 `python -E`、`-I`、`-S`：这三个标志分别忽略 `PYTHONPATH`、启用隔离模式、跳过 `site`，会使启动钩子不被加载。命令中出现它们时 `turbo-physai run` 直接报错。

Shell 语法本身不被解析。`source`、`ulimit`、管道、条件分支和其他系统编排应继续保留在作业脚本中，并由该脚本调用 `turbo-physai run`。

## 6. 失败与停止

- RuntimeConfig 加载或应用失败时，训练在入口执行前失败；
- rank 无法应用 OptimizationConfig 时以退出码 `91` 终止，而不是以未优化状态继续训练；
- `turbo-physai run` 用 `exec` 替换自身，进程树中没有中间进程，`Ctrl+C` 和作业系统的信号直接送达训练进程，停止行为与不使用 `turbo-physai run` 时完全一致。

RuntimeConfig 的修改可能影响通信、CPU 调度和算子选择。交付前应在目标机器上重新验证训练正确性和稳定性能。
