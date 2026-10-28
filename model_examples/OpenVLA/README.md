# OpenVLA 应用说明

## 1. 模型简介

[OpenVLA](https://github.com/openvla/openvla) 是 7B 规模的开源视觉-语言-动作（VLA）模型，基于 Prismatic VLM（DINOv2 + SigLIP 视觉编码器与 Llama-2 7B 语言模型）构建，把机器人操作观测和语言指令映射为离散化的 7 自由度动作 token，支持在 BridgeData V2 等操作数据集上进行全参数微调。训练使用 PyTorch FSDP1 分片，入口脚本为 `vla-scripts/train.py`。

## 2. 优化接入基线

TurboPhysAI 的 OpenVLA 优化基于官方仓库 commit `c8f03f48af692657d3060c19588038c7220e9af9` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。


## 3. 构建训练环境

### 3.1 构建镜像

```bash
docker run -it --name openvla \
--network=host \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
-v /opt/hyhal:/opt/hyhal/:ro \
--privileged=true \
--shm-size 64G \
harbor.sourcefind.cn:5443/hcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10
```

下文命令均在容器内执行。


### 3.2 安装 OpenVLA 训练依赖

镜像已提供 DTK 版 torch 等基础包，OpenVLA 训练还需要安装额外依赖，清单见 [requirements.txt](requirements.txt)。
安装时带 `--force-reinstall --no-deps`，避免 pip 重新解析依赖、把镜像中的 DTK 版 torch 或刚安装的 HCU 版 tensorflow 换成 PyPI 版本。

```bash
# 安装 HCU 版 tensorflow
pip install tensorflow==2.18.0+das.opt1.dtk2604 -i https://pypi.sourcefind.cn/release/dtk/

# 安装其余依赖（--no-deps 避免这些库把 tensorflow 换成其他版本）
pip install --force-reinstall --no-deps -r /path/to/TurboPhysAI/model_examples/OpenVLA/requirements.txt

# 安装 dlimp
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla" --no-deps
```

安装 TurboPhysAI 框架见[安装说明](../../docs/zh/get_started/installation.md)。

## 4. 准备模型源码

```bash
mkdir -p /workspace/model
cd /workspace/model
git clone https://github.com/openvla/openvla.git
cd /workspace/model/openvla
git checkout --detach c8f03f48af692657d3060c19588038c7220e9af9
```


## 5. 准备 BridgeData V2 数据

OpenVLA 训练使用 RLDS 格式的数据。本文使用 BridgeData V2，从官方地址下载（约 124 GB）：

```bash
mkdir -p /workspace/dataset/BridgeDataV2
cd /workspace/dataset/BridgeDataV2

wget -r -nH --cut-dirs=4 --reject="index.html*" \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 目录名必须是 bridge_orig，否则启动后会因找不到数据集而报错
mv bridge_dataset bridge_orig
```

数据根目录下的结构应为：

```text
/workspace/dataset/BridgeDataV2
└── bridge_orig/
    └── 1.0.0/
        ├── dataset_info.json
        └── bridge_orig-train.tfrecord-*
```

启动训练时通过 `--data_root_dir` 传入`/workspace/dataset/BridgeDataV2`。

## 6. 准备基座 VLM 权重

`--vla.type prism-dinosiglip-224px+mx-bridge` 的预训练权重为 `prism-dinosiglip-224px+7b`，训练脚本在未指定 `--pretrained_checkpoint` 时从 Hugging Face Hub 拉取并加载该 Prismatic VLM。按官方说明在仓库根目录准备 Hugging Face token：

```bash
cd /workspace/model/openvla

# 将 Hugging Face token（形如 hf_...）写入openvla仓库根目录的 .hf_token
printf '%s\n' 'hf_xxxxxxxxxxxxxxxxxxxxxxxx' > .hf_token
```

openvla也可以改为加载本地检查点，此时需通过 `--pretrained_checkpoint` 传入本地检查点路径（`--vla.type` 仍需保留，用于选择训练配置）：

```bash
--pretrained_checkpoint <openvla-7b-prismatic 检查点文件或 Prismatic run 目录>
```

## 7. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 OpenVLA 优化。OpenVLA 优化覆盖该训练栈的算子、通信、数据与可复现性环节。

OpenVLA RuntimeConfig 包含训练所需的运行环境和 NUMA 配置。

### 7.1 快速验证（单机八卡）

```bash
# 这里把官方启动命令`--vla.train_strategy fsdp-full-shard`改为fsdp-shard-grad-op策略，减少通信开销
turbo-physai run \
  --model openvla \
  --log-report \
  torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
    --vla.train_strategy fsdp-shard-grad-op \
    --data_root_dir /workspace/dataset/BridgeDataV2 \
    --vla.max_steps 1000 \
    --run_root_dir <训练输出目录> \
    --run_id_note openvla_8card_quick_start \
    --trackers '[jsonl]'
```

### 7.2 单机八卡训练

```bash
# 这里把官方启动命令`--vla.train_strategy fsdp-full-shard`改为fsdp-shard-grad-op策略，减少通信开销
turbo-physai run \
  --model openvla \
  --log-report \
  torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
    --vla.train_strategy fsdp-shard-grad-op \
    --data_root_dir /workspace/dataset/BridgeDataV2 \
    --run_root_dir <训练输出目录> \
    --run_id_note openvla_8card \
    --trackers '[jsonl]'
```

`--model openvla` 自动加载内置 OptimizationConfig 和 RuntimeConfig。

优化应用状态的确认方法见[快速开始：通过日志确认优化状态](../../docs/zh/get_started/quick_start.md#3-通过日志确认优化状态)。
