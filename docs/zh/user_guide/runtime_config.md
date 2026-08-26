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
  torchrun --nproc-per-node=8 tools/train.py path/to/config.py
```

选择规则如下：

1. 指定 `--runtime-config` 时加载该文件；
2. 否则，指定 `--model` 时自动查找模型随包配置；
3. 两者都未指定时，不加载 RuntimeConfig；
4. 文件加载后，`--set` 覆盖环境变量，`--disable-numa` 关闭本次启动的 NUMA 绑定。

未加载 RuntimeConfig 不影响 common OptimizationConfig 的选择。两类配置由 Runner 独立解析。

## 5. 与训练命令的关系

Runner 在执行原训练命令前，将 RuntimeConfig 中的设置写入训练环境。启动链路创建的
后续进程继承该环境。RuntimeConfig 加载或应用失败时，原训练命令不会启动。

训练命令的透传规则、启动条件、参数和返回码见 [CLI 参考](../reference/cli.md)。
RuntimeConfig 的修改可能影响通信、CPU 调度和算子选择，交付前应在目标环境中验证训练
正确性和稳定性能。
