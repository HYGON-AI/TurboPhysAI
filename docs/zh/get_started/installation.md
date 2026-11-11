# 安装指南

TurboPhysAI 支持以下两种部署方式：

- **配套镜像**：面向模型训练用户，镜像提供匹配的 HCU 软件环境，在容器内安装 TurboPhysAI wheel 后使用；
- **源码安装**：面向组件开发、优化接入和问题调试，在已准备好的 HCU 开发环境中构建并安装 TurboPhysAI。

模型源码、数据集和权重不随 TurboPhysAI 配套镜像交付。具体准备方法和训练命令见[模型支持清单](../models/support_list.md)中的对应模型说明。

## 使用配套镜像

配套镜像提供 DTK、PyTorch、FlashAttention、LightOP 和 hipDNN 等基础组件。创建容器后，按下文步骤安装 TurboPhysAI。

使用前需要另外准备：

- 文档指定 commit 的模型源码；
- 处理完成的数据集；
- 模型所需的预训练权重；
- 训练输出目录。

### 获取镜像

根据模型类型设置镜像地址：

自动驾驶模型：

```bash
export IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/autonomous-driving:ubuntu22.04-dtk26.04-torch2.10-py3.10-20261107
```

VLA、DiT 和世界模型：

```bash
export IMAGE=harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10
```

### 创建容器

设置容器名称和宿主机工作目录，然后拉取镜像：

```bash
export CONTAINER_NAME=turbophysai
export WORKSPACE=/path/to/turbophysai-workspace

mkdir -p "$WORKSPACE"
docker pull "$IMAGE"
```

在已配置 HCU 容器运行环境的主机上创建容器：

```bash
docker run -dit \
  --network=host \
  --name="$CONTAINER_NAME" \
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
  --workdir=/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$WORKSPACE":/workspace \
  "$IMAGE" \
  /bin/bash

docker exec -it "$CONTAINER_NAME" bash
```

模型仓库、数据集、权重和训练输出均可放入宿主机的 `$WORKSPACE`，并通过容器内的 `/workspace` 访问。已有数据位于其他目录时，可按需为 `docker run` 增加只读或读写挂载。

### 安装或更新 TurboPhysAI

在容器中，使用系统 Python 安装组件：

```bash
/usr/bin/python3 -m pip install --upgrade turbo-physai \
  -i https://pypi.sourcefind.cn/nightly/dtk2604/ --no-deps
```

## 从源码安装

源码安装仅适用于已经具备完整 HCU 开发环境的场景。构建前应确认以下依赖已安装且版本匹配：

- Linux；
- Python 3.10 或更高版本；
- HCU 驱动和 DTK/ROCm 工具链；
- HCU 版本的 PyTorch；
- hipDNN 1.0.0 或更高版本；
- OptimizationConfig 所需的高性能算子库，例如 LightOp；
- Ninja 和支持 C++17 的编译器；
- 构建所需的 HCU 头文件和系统库。

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install ninja wheel setuptools
MAX_JOBS=8 python setup.py bdist_wheel
python -m pip install --no-deps dist/turbo_physai-*.whl
```

`--no-deps` 仅用于上述依赖已经由当前 HCU 环境提供的情况。它不会安装或校验 PyTorch、hipDNN、LightOp 等平台运行依赖。

`setup.py` 根据 `ROCM_HOME` 或 `ROCM_PATH` 定位工具链。需要限制目标架构时，可在构建命令中设置：

```bash
PYTORCH_ROCM_ARCH=<target-arch> \
MAX_JOBS=8 \
python setup.py bdist_wheel
```

`<target-arch>` 必须替换为目标设备和当前 DTK 支持的 HCU 架构。

## 检查安装结果

使用配套镜像时在容器内执行；使用源码安装时在安装使用的 Python 环境中执行：

```bash
python -c "import torch, hipdnn; print(torch.__version__)"
python -c "import turbo_physai; print(turbo_physai.__file__)"
python -c "import turbo_physai._C; print(turbo_physai._C.__file__)"
turbo-physai --help
```

上述命令全部成功且 `turbo-physai --help` 正常显示帮助信息，表示组件及原生算子扩展可以加载。`turbo_physai._C` 加载失败通常表示编译产物与 PyTorch ABI、DTK/ROCm Runtime 或动态库搜索路径不匹配。

## 卸载组件

```bash
/usr/bin/python3 -m pip uninstall turbo-physai
```

卸载 TurboPhysAI 不会删除模型文件、数据集或 PyTorch 编译缓存。

## 下一步

- 首次运行：[快速开始](quick_start.md)
- 模型选择：[支持模型清单](../models/support_list.md)
- 安装与运行问题：[问题排查](../user_guide/troubleshooting.md)
