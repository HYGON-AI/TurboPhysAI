# 安装指南

TurboPhysAI 支持以下两种部署方式：

- **产品镜像**：面向模型训练用户，镜像内已安装 TurboPhysAI 及匹配的 HCU 软件环境；
- **源码安装**：面向组件开发、优化接入和问题调试，在已准备好的 HCU 开发环境中构建并安装 TurboPhysAI。

模型源码、数据集和权重不随 TurboPhysAI 产品镜像交付。具体准备方法和训练命令见[模型支持清单](../models/support_list.md)中的对应模型说明。

## 使用产品镜像

产品镜像已包含以下内容：

- TurboPhysAI Python 包、命令行工具和原生算子扩展；
- 与目标设备匹配的 HCU Runtime、PyTorch、hipDNN 和高性能算子库；
- 已验证的模型依赖；
- 随组件交付的 OptimizationConfig 和 RuntimeConfig。

使用前需要另外准备：

- 文档指定 commit 的模型源码；
- 处理完成的数据集；
- 模型所需的预训练权重；
- 训练输出目录。

### 获取镜像

从[光源社区 HCU 镜像服务](https://developer.sourcefind.cn/servicelist/detail?post_id=1ad6446c-a4e2-11f1-9669-0242ac150003&active=TagDownload)获取 TurboPhysAI 产品镜像地址。下文使用 `REPOSITORY:TAG` 表示完整镜像名称，请在执行命令前替换为实际值。

### 创建容器

设置容器名称和宿主机工作目录，然后拉取镜像：

```bash
export CONTAINER_NAME=turbophysai
export WORKSPACE=/path/to/turbophysai-workspace

mkdir -p "$WORKSPACE"
docker pull REPOSITORY:TAG
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
  REPOSITORY:TAG \
  /bin/bash

docker exec -it "$CONTAINER_NAME" bash
```

模型仓库、数据集、权重和训练输出均可放入宿主机的 `$WORKSPACE`，并通过容器内的 `/workspace` 访问。已有数据位于其他目录时，可按需为 `docker run` 增加只读或读写挂载。产品镜像已经安装 TurboPhysAI，无需在容器内重复安装。

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

PyTorch、DTK、hipDNN 和 LightOp 等平台依赖应由产品镜像或 HCU 软件源提供。默认 Python 软件源可能不提供这些平台专用包，不能依赖 `pip` 自动安装。

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

使用产品镜像时在容器内执行；使用源码安装时在安装使用的 Python 环境中执行：

```bash
python -c "import torch, hipdnn; print(torch.__version__)"
python -c "import turbo_physai; print(turbo_physai.__file__)"
python -c "import turbo_physai.ops; print(turbo_physai.ops.__file__)"
turbo-physai --help
```

上述命令全部成功且 `turbo-physai --help` 正常显示帮助信息，表示组件及原生算子扩展可以加载。`turbo_physai.ops` 加载失败通常表示编译产物与 PyTorch ABI、DTK/ROCm Runtime 或动态库搜索路径不匹配。

## 卸载源码安装版本

```bash
python -m pip uninstall turbo_physai
```

卸载 TurboPhysAI 不会删除模型文件、数据集或 PyTorch 编译缓存。

## 下一步

- 首次运行：[快速开始](quick_start.md)
- 模型选择：[支持模型清单](../models/support_list.md)
- 安装与运行问题：[问题排查](../user_guide/troubleshooting.md)
