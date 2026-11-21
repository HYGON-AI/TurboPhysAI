# Sparse4D 应用说明

本文说明如何在 autonomous-driving 配套镜像中，基于 Sparse4D 官方仓库接入 TurboPhysAI 的 HCU 优化并启动训练。

## 1. 模型简介

[Sparse4D](https://github.com/HorizonRobotics/Sparse4D) 基于稀疏框架的端到端自动驾驶规划模型。本接入面向官方 Sparse4D 在[NuScenes dataset](https://www.nuscenes.org/nuscenes)数据集上训练配置。

## 2. 优化接入基线

TurboPhysAI 的 Sparse4D 优化基于官方仓库 commit `249ffbb695f4e9db628d953e2bf6d36de04bbb69` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 构建训练环境

### 3.1 创建容器并安装 TurboPhysAI

请先按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 autonomous-driving 配套镜像创建容器并安装 TurboPhysAI, 下文命令均在容器内执行。

### 3.2 安装 Sparse4D 训练依赖

镜像已提供不同mmcv环境包含 HCU 版 PyTorch、Triton 和 FlashAttention等包，Sparse4D训练使用mmcv1环境，启动训练前需要先激活训练环境
```bash
conda activate mmcv1
```
其余训练依赖见 [requirements.txt](requirements.txt)。


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
直接运行Sparse4D源码需要编译deformalbe_aggregation CUDA op, 先查看GPU设备架构：
```bash
rocminfo | grep -o 'gfx[0-9a-z]*' | sort -u
```
执行编译：
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

只使用TurboPhysAI运行优化后的代码，`turbo_physai.operators`中已包含针对HCU优化后的deformalbe_aggregation实现，无需编译，测试能否成功导入。
```bash
cd /workspace
python -c "from turbo_physai.operators import deformable_aggregation_function; print('导入成功:', deformable_aggregation_function)"
```

使用HCU对target.py计算的优化，需要torch linear assignment的实现。下载源码并编译安装torch_linear_assignment包：
```bash
cd /workspace
git clone https://github.com/ivan-chai/torch-linear-assignment.git
cd torch-linear-assignment
git checkout tags/0.0.6
export ROCM_HOME=/opt/dtk
export PATH=$ROCM_HOME/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_HOME/lib:$ROCM_HOME/lib64:$LD_LIBRARY_PATH
# 使用对应 GPU 架构
export PYTORCH_ROCM_ARCH="gfx936"
export TLA_BUILD_CUDA=1 \
export MAX_JOBS=8 \
python setup.py install
python -c "from torch_linear_assignment import batch_linear_assignment; print('导入成功:', batch_linear_assignment)"
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
