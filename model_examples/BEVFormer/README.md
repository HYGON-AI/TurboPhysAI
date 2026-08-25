# BEVFormer 应用说明

本文说明如何通过组件附带产品镜像，在官方 BEVFormer 仓库基线上应用 TurboPhysAI 的 HCU
优化。通用操作流程参见[快速开始](../../docs/zh/get_started/quick_start.md)。

## 1. 模型简介

[BEVFormer](https://github.com/fundamentalvision/BEVFormer) 是面向自动驾驶纯视觉感知的模型，
通过时空 Transformer 从多摄像头图像中学习统一的鸟瞰视角（BEV）表示，用于三维目标检测等感知任务。

## 2. 优化接入基线

TurboPhysAI 的 BEVFormer 优化基于官方仓库 commit
`66b65f3a1f58caf0507cb2a971b9c0e7f842376c` 接入。

## 3. 使用组件附带产品镜像

镜像 `harbor.sourcefind.cn:5443/dcu/admin/base/turbophysai:2.7.1-ubuntu22.04-dtk26.04-py3.10` 已预装 TurboPhysAI、HCU 运行环境及 BEVFormer
训练依赖，并包含经过验证的 OptimizationConfig 和 RuntimeConfig。镜像不包含 BEVFormer
模型源码；用户需要准备上文指定 commit 的官方仓库，并将模型仓库、数据集和训练输出
目录挂载到容器中。容器内无需重新安装 TurboPhysAI 或模型依赖。

```bash
export IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/turbophysai:2.7.1-ubuntu22.04-dtk26.04-py3.10
export MODEL_ROOT=/path/to/BEVFormer
export DATA_ROOT=/data/bevformer-data
export WORK_ROOT=/data/bevformer-work

docker pull "$IMAGE"
mkdir -p "$DATA_ROOT" "$WORK_ROOT"

git -C "$MODEL_ROOT" rev-parse HEAD
```

`MODEL_ROOT` 必须指向官方 BEVFormer 仓库，且当前 commit 必须为
`66b65f3a1f58caf0507cb2a971b9c0e7f842376c`。在已配置 HCU 容器运行环境的
主机上，挂载模型、数据和工作目录启动镜像：

```bash
docker run -dit \
  --network=host \
  --name=turbophysai-bevformer \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=512G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODEL_ROOT":/workspace/model/BEVFormer \
  -v "$DATA_ROOT":/workspace/model/BEVFormer/data \
  -v "$WORK_ROOT":/workspace/model/BEVFormer/work_dirs \
  "$IMAGE"

docker exec -it turbophysai-bevformer bash
```

下文命令均在容器内的 `/workspace/model/BEVFormer` 执行。模型代码来自宿主机挂载的
官方仓库，TurboPhysAI 和运行依赖由镜像提供。

## 4. 准备官方 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download) 获取并接受许可后，下载
nuScenes v1.0 full 数据集和 CAN bus expansion。BEVFormer 的时间序列训练使用 CAN
bus 位姿，因此不能只准备相机文件。解压后，宿主机数据目录应为：

```text
$DATA_ROOT/
├── can_bus/
└── nuscenes/
    ├── maps/
    ├── samples/
    ├── sweeps/
    ├── v1.0-trainval/
    └── v1.0-test/
```

容器内生成官方 BEVFormer temporal info 文件：

```bash
cd /workspace/model/BEVFormer

python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data

ls -lh data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl
```

该命令生成 `nuscenes_infos_temporal_train.pkl` 与
`nuscenes_infos_temporal_val.pkl`。训练配置从 `data/nuscenes/` 读取这两个文件。
同一命令也会处理 `v1.0-test` 并生成测试 info。

## 5. 准备预训练权重

`BEVFormer-base` 配置默认加载 `ckpts/r101_dcn_fcos3d_pretrain.pth`。将官方 BEVFormer 文档指定
的 R101-DCN-FCOS3D 预训练权重下载到该位置：

```bash
cd /workspace/model/BEVFormer
mkdir -p ckpts
wget -O ckpts/r101_dcn_fcos3d_pretrain.pth \
  https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth
```

TurboPhysAI 已安装在镜像的 Python 环境中。后续命令使用 `--model bevformer`
自动选择随组件安装的 OptimizationConfig 和 RuntimeConfig，不依赖固定安装目录。

## 6. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的
[OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和
[RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用
BEVFormer 优化。

BEVFormer RuntimeConfig 包含训练所需的运行环境和 NUMA 配置。

### 6.1 单机八卡训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate bevformer
cd /workspace/model/BEVFormer

turbo-physai run \
  --model bevformer \
  --report-dir ./turbophysai_reports/train-8p \
  -- \
  torchrun --nproc-per-node=8 tools/train.py \
    ./projects/configs/bevformer/bevformer_base.py \
    --launcher pytorch
```

`--nproc-per-node` 指定当前节点的训练进程数。
`--model bevformer` 自动加载内置 OptimizationConfig 和 RuntimeConfig。RuntimeConfig
会为每个 rank 自动选择本地 NUMA node。需要使用自定义交付配置时，通过
`--optimization-config` 和 `--runtime-config` 指定对应文件。

## 7. Group 说明

当前 OptimizationConfig 启用 7 个独立 Group：

| Group                      | 功能                                                                       |
| -------------------------- | -------------------------------------------------------------------------- |
| `bevformer.mdc`          | 使用 HCU MDC 实现替换 MMCV 入口，并优化 Backbone 图像特征提取              |
| `bevformer.msda`         | 替换 FP16/FP32 Multi-Scale Deformable Attention 及其 aliases              |
| `bevformer.geometry_sca` | 优化 DataLoader、BEV 几何处理、Encoder 和 Spatial Cross Attention 数据链路 |
| `bevformer.tsa`          | 优化 Temporal Self Attention 的线性计算路径                                |
| `bevformer.grid_mask`    | 使用设备端张量化方式优化 GridMask                                          |
| `bevformer.compile`      | 对 BEVFormer Layer 和 Decoder 热点使用 `torch.compile` 包装              |
| `bevformer.training`     | 设置 channels-last、fused AdamW、DataLoader worker 和训练运行时参数        |

一个 Group 被阻断（决策为 `block`），或执行失败并成功回滚时，其他不依赖它的 Group 仍可继续应用。回滚失败表示进程状态不再可信，框架会停止后续执行。

## 8. 可选运行参数

这些环境变量由 BEVFormer Replacement 读取，不属于 OptimizationConfig 字段。修改默认值后必须重新做性能和正确性验证。

| 环境变量                                        | 默认值                         | 作用                                  |
| ----------------------------------------------- | ------------------------------ | ------------------------------------- |
| `ENABLE_TORCH_PROFILER`                       | `0`                          | 是否启用指定迭代区间的 Torch Profiler |
| `PROFILER_START_ITER` / `PROFILER_END_ITER` | `50` / `60`                | Profiler 采集区间                     |
| `PROFILER_STOP_AFTER`                         | `1`                          | 采集结束后是否主动停止训练            |
| `TURBO_PHYSAI_DATALOADER_START_METHOD`          | `fork`                       | DataLoader 多进程启动方式             |
| `TURBO_PHYSAI_WORKERS_PER_GPU`                  | 模型配置值，缺省回退为 `8`   | 每卡 DataLoader worker 数             |
| `TURBO_PHYSAI_DISABLE_TORCH_COMPILE`            | `0`                          | 调试时禁用热点编译                    |
| `TURBO_PHYSAI_TORCH_COMPILE_MODE`               | `max-autotune-no-cudagraphs` | 热点编译模式                          |

`PROFILER_LOG_DIR` 可指定 trace 输出目录。Profiler 是调试能力，开启后不用于正式性能统计。

## 9. 如何判断成功

### 9.1 OptimizationReport

rank 0 的 `turbophysai_reports/` 下会生成：

```text
optimization_report-<run_id>.json
optimization_report-<run_id>.md
```

成功应用完整 OptimizationConfig 时，应看到 7 个 Group 的执行状态均为 `applied`，且：

```text
blocked = 0
failed = 0
rolled_back = 0
not_started = 0
```

以 JSON 文件为结构化事实来源。详细字段见 [OptimizationReport](../../docs/zh/user_guide/report.md)。

### 9.2 训练日志

报告成功只表示优化对象已经完成替换，还应继续确认：

- 模型、数据集和权重正常加载；
- 首个训练迭代能够完成；
- loss 没有 NaN/Inf；
- 预热后迭代时间稳定；
- `TORCH_LOGS=recompiles` 下没有持续产生新图。

## 10. 已知限制与排查

- 上文 commit 是 OptimizationConfig 的接入基线。TurboPhysAI 会校验替换目标的源码和
  AST 证据；目标与配置中记录的基线证据不一致时，对应 Optimization Group 不会应用。
- TurboPhysAI 不管理数据集和权重下载。
- Python 异常通常可以从 Traceback 定位到 `turbo_physai/optimizations/models/bevformer/`；段错误或设备运行时致命错误需要结合运行时日志和 core dump 排查。

遇到问题时先保留 JSON 报告和训练 Traceback，再参考[问题排查](../../docs/zh/user_guide/troubleshooting.md)。
