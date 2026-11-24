# LeRobot ACT 模型应用说明

本文介绍在海光 HCU 上的 LeRobot-ACT 依赖安装、资源准备及训练启动。

## 1. 模型简介

[LeRobot ACT](https://github.com/huggingface/lerobot) 是面向机器人操作任务的动作序列预测模型。

## 2. 优化接入基线

TurboPhysAI 的 ACT 优化基于 LeRobot commit `3f2179f3b69708b6ad009b2e7685dd9d05269ee1`。

## 3. 准备训练环境

按照[安装指南](../../docs/zh/get_started/installation.md#使用产品镜像)使用基于DTK2610版本的VLA、DiT 和世界模型配套镜像启动容器并安装 TurboPhysAI。下文命令均在准备好的环境中执行。

## 4. 安装源码及依赖

### 4.1 拉取官方源码

拉取 LeRobot 官方源码：

```bash
cd /workspace &&
test ! -e lerobot && test ! -L lerobot &&
git init lerobot &&
git -C lerobot remote add origin https://github.com/huggingface/lerobot.git &&
git -C lerobot fetch --depth 1 origin 3f2179f3b69708b6ad009b2e7685dd9d05269ee1 &&
git -C lerobot checkout --detach FETCH_HEAD
```

### 4.2 安装依赖

下载 ACT 依赖环境的requirements.txt：

```bash
curl -fL --create-dirs \
  https://raw.githubusercontent.com/HYGON-AI/TurboPhysAI/main/model_examples/LeRobot-ACT/requirements.txt \
  -o /workspace/TurboPhysAI/model_examples/LeRobot-ACT/requirements.txt
```

安装 [requirements.txt](requirements.txt) 中的依赖和 LeRobot 源码：

```bash
env -u PYTHONPATH /usr/local/bin/python3 -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /workspace/TurboPhysAI/model_examples/LeRobot-ACT/requirements.txt

env -u PYTHONPATH /usr/local/bin/python3 -m pip install \
  --no-deps --no-build-isolation -e /workspace/lerobot
```

## 5. 下载数据和权重

### 5.1 下载数据集

设置 HF 镜像地址并下载 [LeRobot 官方数据集](https://huggingface.co/datasets/lerobot/aloha_mobile_cabinet)：

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
env -u PYTHONPATH /usr/local/bin/python3 -c 'from huggingface_hub import snapshot_download; snapshot_download(
    repo_id="lerobot/aloha_mobile_cabinet",
    repo_type="dataset",
    revision="7a752b39f7e69de7e38aee485a6bab07528a061a",
    local_dir="/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet"
)'
```

### 5.2 下载权重

下载 ACT 使用的 ResNet18 预训练骨干权重：

```bash
TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch \
  env -u PYTHONPATH /usr/local/bin/python3 -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(check_hash=True)'
```

## 6. 运行训练

在 `/usr/local/bin/python3` 环境中进行训练。

启动命令中的关键参数：

- `GPU_IDS`：使用的卡号，当前为 8 卡。
- `--batch_size`：每卡的 batch size，当前为 86。
- `--num_workers`：每个训练进程的数据加载 worker 数，当前为 8。
- `--prefetch_factor`：每个 worker 预取的 batch 数，当前为 2。
- `--steps`：训练总步数，当前为 100000。
- `--dataset.repo_id`：数据集在 Hugging Face Hub 上的仓库标识。
- `--dataset.root`：数据集在容器内的本地路径，需与下载数据时的 `local_dir` 一致。
- `--dataset.revision`：指定数据集版本，应与本地数据下载时使用的版本保持一致。
- 日志：命令末尾的 `tee -a` 将训练输出保存到 `/workspace/act_runs/ACT_${GPU_COUNT}gpu_bs${BATCH_SIZE}_${STAMP}.log`。
- 权重：默认启用 checkpoint 保存，每 20000 step 及最后一步自动保存；可通过 `--save_checkpoint` 控制是否保存、`--save_freq` 设置间隔。`--output_dir` 指定输出目录，权重位于该目录下的 `checkpoints/<step>/pretrained_model/model.safetensors`。

编译缓存由 [runtime.yaml](../../turbo_physai/optimizations/models/lerobot_act/configs/runtime.yaml) 设置：

- `TORCHINDUCTOR_CACHE_DIR`：TorchInductor 编译产物的缓存目录，当前配置为 `/workspace/act_runs/inductor_cache`。
- `TRITON_CACHE_DIR`：Triton kernel 编译产物的缓存目录，当前配置为 `/workspace/act_runs/triton_cache`。

```bash
cd /workspace/lerobot
mkdir -p /workspace/act_runs
GPU_IDS=0,1,2,3,4,5,6,7
GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPU_IDS}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=86
env -u PYTHONPATH HIP_VISIBLE_DEVICES="${GPU_IDS}" \
  TORCH_HOME=/workspace/act_assets/ckpt/LeRobot-ACT/torch HF_HUB_OFFLINE=1 \
  turbo-physai run \
    --model lerobot_act \
    --log-report \
    -- \
    torchrun --standalone --nnodes=1 --nproc_per_node="${GPU_COUNT}" \
      -m lerobot.scripts.lerobot_train \
      --policy.type=act \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --dataset.repo_id=lerobot/aloha_mobile_cabinet \
      --dataset.root=/workspace/act_assets/data/LeRobot-ACT/aloha_mobile_cabinet \
      --dataset.revision=7a752b39f7e69de7e38aee485a6bab07528a061a \
      --dataset.video_backend=torchcodec \
      --batch_size="${BATCH_SIZE}" \
      --num_workers=8 \
      --prefetch_factor=2 \
      --steps=100000 \
      --log_freq=1 \
      --wandb.enable=false \
      --output_dir="/workspace/act_runs/turbo_${GPU_COUNT}gpu_bs${BATCH_SIZE}_${STAMP}" \
  2>&1 | tee -a "/workspace/act_runs/ACT_${GPU_COUNT}gpu_bs${BATCH_SIZE}_${STAMP}.log"
```
