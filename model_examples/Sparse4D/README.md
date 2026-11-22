# Sparse4D 应用说明

本文说明如何在 autonomous-driving 配套镜像中，基于 Sparse4D 官方仓库接入 TurboPhysAI 的 HCU 优化并启动训练。

## 1. 模型简介

[Sparse4D](https://github.com/HorizonRobotics/Sparse4D) 基于稀疏框架的端到端自动驾驶规划模型。本接入面向官方 Sparse4D 在[NuScenes dataset](https://www.nuscenes.org/nuscenes)数据集上训练配置。

## 2. 优化接入基线

TurboPhysAI 的 Sparse4D 优化基于官方仓库 commit `249ffbb695f4e9db628d953e2bf6d36de04bbb69` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

### 3.1 创建容器并安装 TurboPhysAI

请先按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 autonomous-driving 配套镜像创建容器，进入容器后激活Sparse4D训练需要环境：

```bash
conda activate mmcv1
```

在`mmcv1`环境中安装 TurboPhysAI, 下文命令均在容器内执行。

### 3.2 安装 Sparse4D 训练依赖

镜像已提供不同mmcv环境包含 HCU 版 PyTorch、Triton 和 FlashAttention等包.

要开启`sparse4d.detection3d_targe`对`target.py`计算的优化，需要torch linear assignment的实现。先查看GPU设备架构：
```bash
rocminfo | grep -o 'gfx[0-9a-z]*' | sort -u
```
确认激活环境，下载源码并编译安装torch_linear_assignment包：
```bash
cd /workspace
conda activate mmcv1
git clone https://github.com/ivan-chai/torch-linear-assignment.git
cd torch-linear-assignment
git checkout tags/0.0.6
export ROCM_HOME=/opt/dtk
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH
# 使用对应 GPU 架构
export PYTORCH_ROCM_ARCH="gfx936"
export TLA_BUILD_CUDA=1
export MAX_JOBS=8
python setup.py install
python -c "from torch_linear_assignment import batch_linear_assignment; print('导入成功:', batch_linear_assignment)"
```

将 `TURBO_PHYSAI_ROOT` 设置为容器内的 TurboPhysAI 仓库路径：

```bash
export TURBO_PHYSAI_ROOT="/workspace/TurboPhysAI"
```


## 4. 准备模型源码
获取Sparse4D源代码，切换到对应分支。

```bash
cd /workspace
git clone https://github.com/HorizonRobotics/Sparse4D.git
cd Sparse4D
git checkout --detach 249ffbb695f4e9db628d953e2bf6d36de04bbb69
git status --short
```

直接运行Sparse4D源码需要编译`deformalbe_aggregation` CUDA 算子：

```bash
cd /workspace/Sparse4D/projects/mmdet3d_plugin/ops
export ROCM_HOME=/opt/dtk
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH
# 使用对应 GPU 架构
export PYTORCH_ROCM_ARCH="gfx936"
export TLA_BUILD_CUDA=1
export MAX_JOBS=8
python3 setup.py develop
```

`turbo_physai.operators`中已包含针对HCU优化后的deformalbe_aggregation实现，使用TurboPhysAI运行优化后的代码，无需上述编译，但需要开启`sparse4d.daf_import_compatibility`、`sparse4d.deformable_aggregation`优化。测试能否成功导入。
```bash
cd /workspace
python -c "from turbo_physai.operators import deformable_aggregation_function; print('导入成功:', deformable_aggregation_function)"
```

## 5. 准备 NuScenes dataset 数据

下载 [NuScenes](https://www.nuscenes.org/nuscenes#download) 数据集并创建符号链接

```bash
cd /workspace
mkdir -p data
ln -s path/to/nuscenes ./data/nuscenes
```

将数据集的元信息和标签进行打包，并生成所需的 .pkl 文件
```bash
mkdir -p /workspace/data/nuscenes_anno_pkls
cd /workspace/Sparse4D
python3 tools/nuscenes_converter.py \
  --root_path /workspace/data/nuscenes \
  --version v1.0-trainval,v1.0-test \
  --info_prefix /workspace/data/nuscenes_anno_pkls/nuscenes
```
通过 K-means 方法生成训练所需anchors
```bash
OPENBLAS_NUM_THREADS=16 \
python3 tools/anchor_generator.py \
	--ann_file /workspace/data/nuscenes_anno_pkls/nuscenes_infos_train.pkl\
    --output_file_name nuscenes_kmeans900.npy
```

## 6. 准备预训练权重

下载 Resnet50 权重, 用于模型训练：

```bash
cd /workspace
mkdir -p ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

镜像使用`torch=2.10`环境，修复数据格式与torch2.10不兼容问题

```bash
cp ckpt/resnet50-19c8e357.pth ckpt/resnet50-19c8e357.pth.bak
python -I -c "
import torch
state = torch.load('/workspace/ckpt/resnet50-19c8e357.pth.bak', map_location='cpu', weights_only=False)
torch.save(state, '/workspace/ckpt/resnet50-19c8e357.pth')
print('converted')
"
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run --model Sparse4D -- <训练命令>` 会从已安装的 TurboPhysAI 包的 `turbo_physai/optimizations/models/Sparse4D/configs/` 目录自动加载 `optimization.yaml` 和 `runtime.yaml`。`--` 后传入原始训练命令，`--log-report` 等 TurboPhysAI 选项放在 `--` 前。

### 7.1 单机八卡训练

`run.sh` 基于官方 [`tools/train.py`](../../../Sparse4D/tools/train.py) 的单机八卡启动方式，脚本已包含 `--model Sparse4D` 和 `--log-report`，可直接运行。

```bash
bash "$TURBO_PHYSAI_ROOT/model_examples/Sparse4D/run.sh"
```

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
