# FlashOCC 应用说明

本文说明如何基于官方 FlashOCC 仓库使用 TurboPhysAI 的 HCU 优化。按照[安装指南](../../docs/zh/get_started/installation.md#%E4%BD%BF%E7%94%A8%E9%85%8D%E5%A5%97%E9%95%9C%E5%83%8F)，使用自动驾驶配套镜像创建容器并安装 TurboPhysAI。

## 1. 模型简介

[FlashOCC](https://github.com/Yzichen/FlashOCC) 是面向自动驾驶场景的相机视觉占用预测模型，可从多摄像头图像预测场景中的三维语义占用。

## 2. 优化接入基线

本接入基于官方 FlashOCC 仓库 commit `449dc7947d6549ccb72a99d6f6d8c02f166ec0dd`。

优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备运行环境与源码

### 3.1 设置工作目录

进入容器后执行：

```bash
mkdir "/workspace/FlashOCC_TurboPhysAI"
cd "/workspace/FlashOCC_TurboPhysAI"
```

### 3.2 拉取官方 FlashOCC 源码

```bash
git clone https://github.com/Yzichen/FlashOCC.git "/workspace/FlashOCC_TurboPhysAI/FlashOCC"
git -C "/workspace/FlashOCC_TurboPhysAI/FlashOCC" checkout --detach \
  449dc7947d6549ccb72a99d6f6d8c02f166ec0dd
```

### 3.3 准备 MMDetection3D 依赖并构建 HCU 扩展

```bash
git clone --branch v1.0.0rc4 \
  https://github.com/open-mmlab/mmdetection3d.git \
  "/workspace/FlashOCC_TurboPhysAI/FlashOCC/mmdetection3d"
git -C "/workspace/FlashOCC_TurboPhysAI/FlashOCC/mmdetection3d" checkout --detach \
  c9541b0db89498fdea5cafd05b7b17f7b625b858

source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1

export PYTHONPATH="/workspace/FlashOCC_TurboPhysAI/FlashOCC/mmdetection3d:/workspace/FlashOCC_TurboPhysAI/FlashOCC${PYTHONPATH:+:$PYTHONPATH}"
cd "/workspace/FlashOCC_TurboPhysAI/FlashOCC/projects"
FORCE_CUDA=1 MAX_JOBS=8 python -m pip install -e . --no-build-isolation --no-deps
```

## 4. 准备 nuScenes 数据并生成标注

按照 [FlashOCC 官方安装文档 Step 3](https://github.com/Yzichen/FlashOCC/blob/master/doc/install.md) 下载并准备 nuScenes。将数据放到 `/workspace/FlashOCC_TurboPhysAI/FlashOCC/data/nuscenes`，至少包括以下目录：

```text
/workspace/FlashOCC_TurboPhysAI/FlashOCC/data/nuscenes/
├── v1.0-trainval/
├── sweeps/
├── samples/
├── gts/
└── maps/
```

在 FlashOCC 根目录运行官方数据转换脚本，生成训练所需的 PKL 标注：

```bash
cd /workspace/FlashOCC_TurboPhysAI/FlashOCC && python -c 'import runpy, sys, types; tools = types.ModuleType("tools"); tools.__path__ = ["./tools"]; sys.modules["tools"] = tools; runpy.run_path("tools/create_data_bevdet.py", run_name="__main__")'
```

训练脚本读取以下标注文件：

```text
/workspace/FlashOCC_TurboPhysAI/FlashOCC/data/nuscenes/
├── bevdetv2-nuscenes_infos_train.pkl
└── bevdetv2-nuscenes_infos_val.pkl
```

## 5. 准备预训练权重

从 [BEVDet 官方模型列表](https://github.com/HuangJunJie2017/BEVDet/blob/dev3.0/README.md#nuscenes-detection) 的 `BEVDet-R50-CBGS` 下载权重，并将其放到训练脚本使用的 `/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth`：

```bash
mkdir -p "/workspace/FlashOCC_TurboPhysAI/static_data/ckpts"
```

准备完成后的主要目录如下：

```text
/workspace/FlashOCC_TurboPhysAI/
├── FlashOCC/
│   ├── data/nuscenes/
│   └── mmdetection3d/
└── static_data/
```

## 6. 启动训练

以下命令通过 `--set` 指定当前容器的模型路径、设备和单机网卡。默认使用单机8卡、每卡 batch size 4和每卡 24 个数据加载 worker。多机运行时需要修改NCCL_SOCKET_IFNAME和GLOO_SOCKET_IFNAME。

通过 `turbo-physai run --model flashocc` 加载模型配置并启动训练：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1

cd "/workspace/FlashOCC_TurboPhysAI/FlashOCC"

turbo-physai run \
  --model flashocc \
  --set HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --set NCCL_SOCKET_IFNAME=lo \
  --set GLOO_SOCKET_IFNAME=lo \
  --set FLASHOCC_ROOT=/workspace/FlashOCC_TurboPhysAI/FlashOCC \
  --log-report \
  -- \
  torchrun --nproc_per_node=8 tools/train.py \
    ./projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "/workspace/FlashOCC_TurboPhysAI/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)" \
    --cfg-options \
      data.samples_per_gpu=4 \
      data.workers_per_gpu=24 \
      "load_from=/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth"
```

NUMA 配置、优化应用结果及其检查方法见 [快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。

## 兼容性说明

FlashOCC 的 `ray_metrics.py` 在导入时加载 DVR 扩展，上游编译方式在 HCU 上不兼容。TurboPhysAI 的 FlashOCC 兼容层拦截名为 `dvr` 的 `torch.utils.cpp_extension.load` 调用，过滤 HCU 编译器不支持的参数，并在实际调用 DVR 接口时使用随包提供的兼容源码延迟编译；其他扩展加载不受影响。该兼容处理不属于优化组，不能通过 `--disable-group` 关闭，也不会出现在 OptimizationReport 中。

FlashOCC 的 `bevdet_occ.py` 包含 GB18030 编码的源码，引擎通过 `inspect.getsource` 读取源码时可能失败，导致 `forward_occ_train` 无法进行哈希信任检查。当前兼容层仅针对该文件在需要时使用 GB18030 解码并填充 `linecache`，使引擎能够读取源码；后续将由引擎提供通用源码编码处理后移除该兼容层。
