# OpenVLA 应用说明

本文说明如何在产品镜像中，基于官方 OpenVLA 仓库应用 TurboPhysAI 的 HCU 优化。

## 1. 模型简介

[OpenVLA](https://github.com/openvla/openvla) 是 7B 规模的开源视觉-语言-动作（VLA）模型，基于 Prismatic VLM（DINOv2 + SigLIP 视觉编码器与 Llama-2 7B 语言模型）构建，把机器人操作观测和语言指令映射为离散化的 7 自由度动作 token，支持在 BridgeData V2 等操作数据集上进行全参数微调。

训练使用 PyTorch FSDP1 分片，入口脚本为 `vla-scripts/train.py`。TurboPhysAI 的 OpenVLA 优化覆盖该训练栈的算子、通信、数据与可复现性环节，全部通过随包 OptimizationConfig 和 RuntimeConfig 交付，不修改模型源码。

## 2. 优化接入基线

TurboPhysAI 的 OpenVLA 优化基于官方仓库 commit `c8f03f48af692657d3060c19588038c7220e9af9` 接入。优化接入基线的使用建议见[模型支持清单](../../docs/zh/models/support_list.md)。

## 3. 准备模型源码

```bash
cd /workspace
mkdir -p model
git clone https://github.com/openvla/openvla.git model/OpenVLA
cd model/OpenVLA
git checkout --detach c8f03f48af692657d3060c19588038c7220e9af9
```

也可以将准备好的官方 OpenVLA 仓库放入宿主机工作目录的 `model/OpenVLA`。下文命令均在容器内的 `/workspace/model/OpenVLA` 执行。产品镜像提供 TurboPhysAI 与 HCU 运行环境，不包含模型源码；OpenVLA 训练依赖（`torch`、`transformers`、`timm`、`flash-attn`、RLDS/dlimp、`draccus` 等）按[官方安装说明](https://github.com/openvla/openvla#installation)在该仓库的训练环境中准备。

## 4. 准备 BridgeData V2 数据

OpenVLA 训练使用 RLDS 格式的数据。本文使用 BridgeData V2，从官方地址下载（约 124 GB）：

```bash
cd <BridgeData V2 数据根目录>

wget -r -nH --cut-dirs=4 --reject="index.html*" \
  https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# 目录名必须是 bridge_orig，否则启动后会因找不到数据集而报错
mv bridge_dataset bridge_orig
```

数据根目录下的结构应为：

```text
<BridgeData V2 数据根目录>/
└── bridge_orig/
    └── 1.0.0/
        ├── dataset_info.json
        └── bridge_orig-train.tfrecord-*
```

启动训练时通过 `--data_root_dir` 传入该数据根目录。TurboPhysAI 不下载、转换或重新分发数据集。

## 5. 准备基座 VLM 权重

`--vla.type prism-dinosiglip-224px+mx-bridge` 的 `base_vlm` 为 `prism-dinosiglip-224px+7b`，训练脚本在未指定 `--pretrained_checkpoint` 时从 Hugging Face Hub 拉取该 Prismatic VLM。按官方说明在仓库根目录准备 Hugging Face token：

```bash
cd /workspace/model/OpenVLA

# 将 Hugging Face token（形如 hf_...）写入仓库根目录的 .hf_token
printf '%s\n' 'hf_xxxxxxxxxxxxxxxxxxxxxxxx' > .hf_token
```

离线环境可以先在有网络的机器上填充 Hugging Face 缓存，再把缓存目录挂载进容器并通过 `HF_HOME` 指向它。也可以改为加载本地检查点，此时 `--pretrained_checkpoint` 取代上面的 Hub 下载路径（`--vla.type` 仍需保留，用于选择训练配置）：

```bash
--pretrained_checkpoint <openvla-7b-prismatic 检查点文件或 Prismatic run 目录>
```

## 6. 通过 turbo-physai run 启动训练

`turbo-physai run` 自动加载随包交付的 [OptimizationConfig](../../docs/zh/user_guide/optimization_config.md) 和 [RuntimeConfig](../../docs/zh/user_guide/runtime_config.md)，无需修改模型源码即可应用 OpenVLA 优化。

### 6.1 单机八卡训练

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate <OpenVLA 训练环境>
cd /workspace/model/OpenVLA

turbo-physai run \
  --model openvla \
  --log-report \
  torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
    --vla.type prism-dinosiglip-224px+mx-bridge \
    --vla.train_strategy fsdp-shard-grad-op \
    --vla.expected_world_size 8 \
    --vla.global_batch_size 256 \
    --vla.per_device_batch_size 32 \
    --vla.enable_gradient_checkpointing true \
    --vla.reduce_in_full_precision true \
    --vla.max_steps 1000 \
    --data_root_dir <BridgeData V2 数据根目录> \
    --run_root_dir <训练输出目录> \
    --run_id_note openvla_8card \
    --trackers '[jsonl]'
```

需要使用自定义交付配置时，通过 `--optimization-config` 和 `--runtime-config` 指定对应文件。
