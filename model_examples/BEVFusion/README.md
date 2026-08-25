# BEVFusion 应用说明

本文说明如何通过组件附带产品镜像，在官方 BEVFusion 仓库基线上应用 TurboPhysAI 的 HCU
优化。通用操作流程参见[快速开始](../../docs/zh/get_started/quick_start.md)。

## 1. 模型简介

[BEVFusion](https://github.com/mit-han-lab/bevfusion) 是面向自动驾驶多传感器感知的融合模型，
在统一的鸟瞰视角（BEV）空间融合相机与激光雷达特征，支持三维目标检测和 BEV 地图分割等任务。

## 2. 优化接入基线

TurboPhysAI 的 BEVFusion 优化基于官方仓库 commit
`326653dc06e0938edf1aae7d01efcd158ba83de5` 接入。

## 3. 使用组件附带产品镜像

镜像 `harbor.sourcefind.cn:5443/dcu/admin/base/turbophysai:2.7.1-ubuntu22.04-dtk26.04-py3.10` 已预装 TurboPhysAI、HCU 运行环境及 BEVFusion
训练依赖，并包含经过验证的 OptimizationConfig 和 RuntimeConfig。镜像不包含 BEVFusion
模型源码；用户需要准备上文指定 commit 的官方仓库，并将模型仓库、数据集、预训练权重
和训练输出目录挂载到容器中。

```bash
export IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/turbophysai:2.7.1-ubuntu22.04-dtk26.04-py3.10
export MODEL_ROOT=/path/to/bevfusion
export DATA_ROOT=/data/bevfusion-data
export WEIGHT_ROOT=/data/bevfusion-weights
export WORK_ROOT=/data/bevfusion-work

docker pull "$IMAGE"
mkdir -p "$DATA_ROOT" "$WEIGHT_ROOT" "$WORK_ROOT"

git -C "$MODEL_ROOT" rev-parse HEAD
```

`MODEL_ROOT` 必须指向官方 BEVFusion 仓库，且当前 commit 必须为
`326653dc06e0938edf1aae7d01efcd158ba83de5`。在已配置 HCU 容器运行环境的
主机上，挂载模型、数据、权重和工作目录启动镜像：

```bash
docker run -dit \
  --network=host \
  --name=turbophysai-bevfusion \
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
  -v "$MODEL_ROOT":/workspace/model/BEVFusion \
  -v "$DATA_ROOT":/workspace/model/BEVFusion/anno_pkls \
  -v "$WEIGHT_ROOT":/workspace/model/BEVFusion/ckpts \
  -v "$WORK_ROOT":/workspace/model/BEVFusion/runs \
  "$IMAGE"

docker exec -it turbophysai-bevfusion bash
```

下文命令均在容器内的 `/workspace/model/BEVFusion` 执行。TurboPhysAI 和运行依赖由镜像
提供；模型源码来自宿主机挂载的官方仓库。

## 4. 准备模型源码

首次使用挂载的官方仓库时，在容器内安装 BEVFusion 源码包及其原生扩展：

```bash
cd /workspace/model/BEVFusion
python setup.py develop
```

## 5. 准备 nuScenes 数据

从 [nuScenes 下载页](https://www.nuscenes.org/download) 获取并接受许可后，按照
[BEVFusion 官方数据准备说明](https://github.com/mit-han-lab/bevfusion/blob/main/docs/prepare_dataset.md)
准备 nuScenes 数据、地图扩展和训练所需的 info 文件。本文命令将准备完成的数据目录挂载为
`/workspace/model/BEVFusion/anno_pkls`，启动训练时通过 `--dataset_root ./anno_pkls/`
传入该路径。

数据目录必须同时包含预处理生成的元数据，以及元数据所引用的相机、激光雷达和地图文件。
启动训练前可先确认：

```bash
cd /workspace/model/BEVFusion
find anno_pkls -maxdepth 2 -type f | head
```

TurboPhysAI 不下载、转换或重新分发 nuScenes 数据。

## 6. 准备预训练权重

Camera-LiDAR 配置使用 Swin Transformer 相机 Backbone 预训练权重和 LiDAR-only
检测权重。将 BEVFusion 官方文档提供的权重放入以下位置：

```text
ckpts/
├── swint-nuimages-pretrained.pth
└── lidar-only-det.pth
```

也可以执行官方仓库提供的下载脚本，再将对应文件复制或链接到 `ckpts/`：

```bash
cd /workspace/model/BEVFusion
bash tools/download_pretrained.sh
mkdir -p ckpts
ln -sf ../pretrained/swint-nuimages-pretrained.pth \
  ckpts/swint-nuimages-pretrained.pth
ln -sf ../pretrained/lidar-only-det.pth \
  ckpts/lidar-only-det.pth
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的
[OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和
[RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用
BEVFusion 优化。

### 7.1 单机八卡训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate bevfusion
cd /workspace/model/BEVFusion

turbo-physai run \
  --model bevfusion \
  --report-dir ./turbophysai_reports/train-8p \
  -- \
  torchpack dist-run -np 8 \
  python tools/train.py \
    configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
    --run-dir runs/turbo_physai_8card \
    --dataset_root ./anno_pkls/ \
    --model.encoders.camera.backbone.init_cfg.checkpoint \
      ckpts/swint-nuimages-pretrained.pth \
    --load_from ckpts/lidar-only-det.pth \
    --fp16 None
```

`torchpack dist-run -np` 指定当前节点的训练进程数。
需要覆盖 RuntimeConfig 中的环境变量时，可在 `--` 前追加 `--set KEY=VALUE`；需要使用
自定义交付配置时，通过 `--optimization-config` 和 `--runtime-config` 指定文件。

## 8. Group 说明

当前 OptimizationConfig 启用 17 个 Group，其中 7 个为公共优化，10 个为 BEVFusion
模型优化。

### 8.1 公共优化

| Group | 功能 |
| --- | --- |
| `mmcv.msda` | 替换 MMCV Multi-Scale Deformable Attention 底层入口 |
| `mmdet3d.gaussian` | 优化二维 Gaussian 生成 |
| `mmdet3d.bev_pool` | 使用 HCU 实现替换 BEV Pool 算子 |
| `mmdet3d.quick_cumsum` | 优化 BEV Pool 的 QuickCumsum 前向与反向计算 |
| `mmdet3d.voxelization` | 使用 HCU 实现替换 Voxelization 入口 |
| `mmdet3d.canonical_indice_pairs` | 优化 SparseConv indice pair 生成 |
| `mmdet3d.sparse_tensor` | 优化 SparseConvTensor 稀疏度计算 |

### 8.2 模型优化

| Group | 功能 |
| --- | --- |
| `bevfusion.import_compatibility` | 处理当前基线的 FlashAttention、可选扩展和 SparseConv Registry 兼容性 |
| `bevfusion.backbone` | 优化相机 Backbone 特征提取和多模态特征组织 |
| `bevfusion.loss_reduction` | 优化训练 loss 汇总路径 |
| `bevfusion.training` | 配置 channels-last、数据预取和模型编译入口 |
| `bevfusion.gaussian` | 编译 Gaussian heatmap 和 radius 热点 |
| `bevfusion.depth_factorization` | 优化 Depth LSS、BEV Pool 及相关张量计算 |
| `bevfusion.bev_geometry` | 优化视角变换的几何坐标计算 |
| `bevfusion.hungarian_transfer` | 优化 TransFusion target 构造、loss 和 Hungarian 分配路径 |
| `bevfusion.transfusion_bbox_coder` | 优化 TransFusionBBoxCoder 构造和调用入口 |
| `bevfusion.compile` | 对 SparseEncoder、ConvFuser、LSS Neck 和 TransFusion 热点应用编译优化 |

一个 Group 被阻断（决策为 `block`），或执行失败并成功回滚时，其他不依赖它的 Group
仍可继续应用。回滚失败表示进程状态不再可信，框架会停止后续执行。

## 9. 可选运行参数

下列参数用于诊断或调整 BEVFusion 模型优化。修改默认值后应重新验证正确性和性能。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `TURBO_PHYSAI_DATALOADER_START_METHOD` | `fork` | DataLoader 多进程启动方式 |
| `TURBO_PHYSAI_DISABLE_TORCH_COMPILE` | `0` | 调试时禁用模型热点编译 |
| `TURBO_PHYSAI_TORCH_COMPILE_MODE` | `max-autotune-no-cudagraphs` | 模型热点编译模式 |
| `MMDET3D_CHANNELS_LAST` | `1` | 启用相机 Backbone channels-last 布局 |
| `MMDET3D_COMPILE_TARGET` | `camera` | 选择训练包装器中的编译目标 |
| `MMDET3D_COMPILE_MODE` | `max-autotune-no-cudagraphs` | 训练包装器使用的编译模式 |
| `MMDET3D_DISABLE_BEV_POOL_PREPARE_OPT` | `0` | 调试时禁用 BEV Pool 输入准备优化 |
| `MMDET3D_DISABLE_BEV_POOL_GEOMETRY_OPT` | `0` | 调试时禁用 BEV 几何优化 |
| `MMDET3D_BEV_POOL_GEOMETRY_BOUNDARY_EPS` | `1e-3` | BEV 几何边界修正阈值 |
| `MMDET3D_BEV_POOL_GEOMETRY_CORRECTION_CHUNK` | `262144` | BEV 几何边界修正分块大小 |

正式交付的默认值由随包 RuntimeConfig 设置。命令行 `--set` 仅用于经过验证的部署调整
或问题定位。

## 10. 如何判断成功

### 10.1 OptimizationReport

rank 0 的报告目录下会生成：

```text
optimization_report-<run_id>.json
optimization_report-<run_id>.md
```

成功应用完整 OptimizationConfig 时，应看到 17 个 Group 的执行状态均为 `applied`，且：

```text
blocked = 0
failed = 0
rolled_back = 0
not_started = 0
```

以 JSON 文件为结构化事实来源。详细字段见
[OptimizationReport](../../docs/zh/user_guide/report.md)。

### 10.2 训练日志

报告成功表示优化对象已经完成替换。训练日志还应满足：

- 模型、数据集和预训练权重正常加载；
- 首个训练迭代能够完成；
- loss 没有 NaN/Inf；
- 预热后迭代时间稳定；
- `TORCH_LOGS=recompiles` 下没有持续产生新图。

## 11. 已知限制与排查

- 上文 commit 是 OptimizationConfig 的接入基线。TurboPhysAI 会校验替换目标的源码和
  AST 证据；目标与配置中记录的基线证据不一致时，对应 Optimization Group 不会应用。
- TurboPhysAI 不管理 nuScenes 数据和预训练权重的下载、许可及存储。
- BEVFusion 源码及其必要原生扩展需要在组件附带产品镜像中正确安装。
- Python 异常通常可以从 Traceback 定位到 `turbo_physai/optimizations/models/bevfusion/`；
  段错误或设备运行时致命错误需要结合运行时日志和 core dump 排查。

遇到问题时先保留 JSON 报告和训练 Traceback，再参考
[问题排查](../../docs/zh/user_guide/troubleshooting.md)。
