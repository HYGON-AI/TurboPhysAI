# Cosmos3 运行指南

在 `cosmos-framework` 根目录应用 Cosmos3 补丁，再直接启动训练。以下示例使用 8 张 HCU，训练 70 步。

验证状态：Generator Nano 已完成训练流程验证；Action Policy 和 Reasoner 提供适配代码与运行命令，待完成训练验证。

## 环境准备

### 拉取镜像及创建容器
```
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10

docker run -it --network=host --name=cosmos_hygon --privileged --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=512G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1  -v /opt/hyhal:/opt/hyhal:ro  harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10-20260929
```
### 准备代码
```bash
git clone https://github.com/HYGON-AI/TurboPhysAI.git
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
git checkout 9726697a
```

### 应用补丁

在 Cosmos 仓库根目录应用补丁；以下假设 TurboPhysAI 和 cosmos-framework 是同级相邻目录：

```bash
git apply --check ../TurboPhysAI/model_examples/Cosmos3/cosmos-framework.patch
git apply ../TurboPhysAI/model_examples/Cosmos3/cosmos-framework.patch
```

如需撤销补丁：

```bash
git apply -R ../TurboPhysAI/model_examples/Cosmos3/cosmos-framework.patch
```

### 安装训练依赖

在 `cosmos-framework` 根目录执行：

```bash
pip install -r ../TurboPhysAI/model_examples/Cosmos3/requirements.txt
pip install lerobot --no-deps
```

## 数据与模型权重

数据与权重沿用上游的官方来源及下载、处理步骤：参见 [训练与数据准备](https://github.com/NVIDIA/cosmos-framework/blob/9726697a83315540c6885baefd2fe353d9c74920/docs/training.md)、[DROID 数据准备](https://github.com/NVIDIA/cosmos-framework/blob/9726697a83315540c6885baefd2fe353d9c74920/docs/action_policy_droid_posttrain.md#full-reproduction) 和 [模型权重准备](https://github.com/NVIDIA/cosmos-framework/blob/9726697a83315540c6885baefd2fe353d9c74920/docs/setup.md#downloading-base-checkpoints)。准备好以下资产后，将后文的 `/path/to/...` 替换为实际路径；DROID 小规模测试见[快速验证（可选）](#快速验证可选)。

| 任务 | 数据 | 模型权重 |
| --- | --- | --- |
| Generator Nano | BridgeData2 的 `sft_dataset_bridge` | Cosmos3-Nano DCP、Wan2.2 VAE、Cosmos3-Nano processor/tokenizer |
| Action Policy | DROID LeRobot 数据集 | Cosmos3-Nano DCP、Wan2.2 VAE、Qwen3-VL-8B tokenizer 缓存 |
| Reasoner | VideoPhy2 或 LLaVA-OV 配方对应数据 | Qwen3-VL 模型目录、Cosmos3-Nano-VLM safetensors |

将 Cosmos3-Nano HF 权重转换为 DCP 时，在 Cosmos 仓库根目录设置 `WAN_VAE_PATH` 和 `BASE_CHECKPOINT_PATH`，然后执行：

```bash
WAN_VAE_PATH="$WAN_VAE_PATH" HF_HUB_OFFLINE=1 \
python -m cosmos_framework.scripts.convert_model_to_dcp \
    -i /path/to/Cosmos3-Nano -o "$BASE_CHECKPOINT_PATH"
```

上游转换脚本会读取 VAE/processor；离线环境需预先准备 Hugging Face 缓存。

## 启动训练

所有命令均在 `cosmos-framework` 根目录执行，训练场景由 `--sft-toml` 选择。
运行环境由 Shell 明确设置，无需 Turbo runtime 配置：

```bash
export PYTHONPATH="$PWD"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1
export HF_HOME=/path/to/hf-cache
export MIOPEN_FIND_MODE=1 PYTORCH_MIOPEN_SUGGEST_NDHWC=1
export TORCH_NCCL_HIGH_PRIORITY=1
```

### Generator Nano

设置数据与权重路径后运行：

```bash
source /opt/MacCodecSDK/env.sh
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATASET_PATH=/path/to/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export LOCAL_PROCESSOR_DIR=/path/to/Cosmos3-Nano
export IMAGINAIRE_OUTPUT_ROOT=./outputs/hcu_training

torchrun --nproc-per-node=8 --master_port=29712 \
    -m cosmos_framework.scripts.train \
    --sft-toml=examples/toml/sft_config/vision_sft_nano.toml \
    -- \
    job.name=generator_vision_sft_nano_hcu_perf_8card \
    job.wandb_mode=offline \
    model.config.vlm_config.tokenizer.pretrained_model_name=$LOCAL_PROCESSOR_DIR \
    dataloader_train.dataloader.prefetch_factor=16 \
    trainer.max_iter=70 \
    checkpoint.save_iter=1000 \
    trainer.callbacks.compile_tokenizer.enabled=true \
    trainer.callbacks.compile_tokenizer.compile_after_iterations=3 \
    'trainer.callbacks.compile_tokenizer.warmup_resolutions=["256"]'
```

### Action Policy（DROID）

```bash
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export LOCAL_PROCESSOR_DIR=/path/to/Cosmos3-Nano
export DROID_ROOT=/path/to/droid_plus_lerobot_640x360_20260412
export KEEP_RANGES_PATH=/path/to/keep_ranges_1_0_1.json
export IMAGINAIRE_OUTPUT_ROOT=./outputs/action_policy
export TORCHINDUCTOR_WORKER_START=spawn TORCHINDUCTOR_COMPILE_THREADS=4
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=8 --master_port=29716 \
  -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/action_policy_droid_nano.toml -- \
  job.wandb_mode=disabled job.name=action_policy_sft_nano_hcu_perf_8card_ops \
  model.config.vlm_config.tokenizer.pretrained_model_name=$LOCAL_PROCESSOR_DIR \
  model.config.parallelism.data_parallel_shard_degree=8 \
  model.config.parallelism.data_parallel_replicate_degree=1 \
  dataloader_train.max_samples_per_batch=8 \
  dataloader_train.dataloader.batch_size=8 \
  dataloader_train.dataloader.num_workers=1 \
  +dataloader_train.dataloader.multiprocessing_context=spawn \
  dataloader_train.dataloader.prefetch_factor=4 \
  dataloader_train.dataloader.datasets.droid.dataset.mode=wam \
  dataloader_train.dataloader.datasets.droid.dataset.video_backend=pyav \
  dataloader_train.dataloader.datasets.droid.dataset.use_filter_dict=true \
  dataloader_train.dataloader.datasets.droid.dataset.filter_dict_path="$KEEP_RANGES_PATH" \
  model.config.rectified_flow_training_config.loss_scale=10.0 \
  trainer.seed=42 trainer.max_iter=70 \
  model.config.compile.enabled=true model.config.compile.compile_dynamic=true \
  trainer.callbacks.compile_tokenizer.enabled=false \
  checkpoint.save_iter=1000 checkpoint.dcp_async_mode_enabled=false
```

`KEEP_RANGES_PATH` 指向上游流程准备的过滤文件；快速验证时替换为 perf64 对应文件。`MIOPEN_FIND_MODE=1` 由前面的 Shell 设置提供。

一次性性能运行如需关闭 checkpoint 写入，追加 `checkpoint.type._target_=cosmos_framework.checkpoint.load_only.LoadOnlyDistributedCheckpointer`；默认保留正常保存行为。

### Reasoner VideoPhy2

```bash
export VIDEOPHYSICS_ROOT=/path/to/videophysics_official
export VLM_MODEL_NAME=/path/to/Qwen3-VL-8B-Instruct
export VLM_SAFETENSORS_PATH=/path/to/Cosmos3-Nano-VLM
export IMAGINAIRE_OUTPUT_ROOT=./outputs/reasoner

torchrun --nproc_per_node=8 --master_port=29711 \
  -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/videophy2_sft_nano.toml -- \
  job.wandb_mode=disabled job.name=reasoner_videophy2_nano_hcu \
  model.config.policy.backbone.model_name="$VLM_MODEL_NAME" \
  model.config.policy.backbone.safetensors_path="$VLM_SAFETENSORS_PATH" \
  model.config.parallelism.data_parallel_shard_degree=8 \
  trainer.seed=42 trainer.max_iter=70 scheduler.cycle_lengths=[70] \
  trainer.callbacks.log_tensor_shape.num_log=-1 checkpoint.save_iter=1000 \
  model.config.compile.enabled=true model.config.compile.compile_dynamic=true \
  model.config.activation_checkpointing.mode=none \
  model.config.parallelism.fsdp_layers_per_group=8
```

### Reasoner LLaVA-OV

沿用上面的 VLM 路径。

```bash
HF_HUB_OFFLINE=0 torchrun --nproc_per_node=8 --master_port=29715 \
  -m cosmos_framework.scripts.train \
  --sft-toml=examples/toml/sft_config/llava_ov.toml -- \
  job.wandb_mode=disabled job.name=reasoner_llava_ov_hcu \
  model.config.policy.backbone.model_name="$VLM_MODEL_NAME" \
  model.config.policy.backbone.safetensors_path="$VLM_SAFETENSORS_PATH" \
  model.config.parallelism.data_parallel_shard_degree=8 data_setting.max_tokens=16000 \
  trainer.seed=42 trainer.max_iter=70 scheduler.cycle_lengths=[70] \
  trainer.callbacks.log_tensor_shape.num_log=-1 \
  checkpoint.save_iter=1000 checkpoint.dcp_async_mode_enabled=false
```

## 快速验证（可选）

Action Policy 可从官方 [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) 生成前 64 个片段的 perf64 子集，无需下载完整数据集。以下固定数据版本，仅下载 metadata 和所需的 8 个数据、视频文件，共约 1.7 GiB。

在 `cosmos-framework` 根目录执行：

```bash
hf download nvidia/Cosmos3-DROID --repo-type dataset \
  --revision 5c11a20accb11497270a5247a7f1e66ad04c956c \
  --local-dir /path/to/droid-official/droid_plus_lerobot_640x360_20260412 \
  --include "success/meta/*" "success/data/chunk-000/file-000.parquet" \
  "success/videos/observation.image.exterior_image_[12]_left/chunk-000/file-00[0-2].mp4" \
  "success/videos/observation.image.wrist_image_left/chunk-000/file-000.mp4"

python examples/prepare_droid_perf_subset.py \
  --source-success /path/to/droid-official/droid_plus_lerobot_640x360_20260412/success \
  --output-root /path/to/perf64/droid_plus_lerobot_640x360_20260412 \
  --episodes 64
```

输出目录直接命名为 `droid_plus_lerobot_640x360_20260412`，供上游 loader 识别数据版本，无需建立软链接。将 Action Policy 训练小节中的数据路径替换为：

```bash
export DROID_ROOT=/path/to/perf64/droid_plus_lerobot_640x360_20260412
export KEEP_RANGES_PATH="$DROID_ROOT/keep_ranges_perf64.json"
```

metadata 和共享数据文件可能包含未选片段，验证时必须使用生成的 `keep_ranges_perf64.json`。准备好数据后，复用前文 Action Policy 的 70 步训练命令。

Generator 的 BridgeData2、Reasoner 的 VideoPhy2 和 Qwen3-VL tokenizer 均按前文链接的上游官方流程准备，模型权重同样沿用上游流程。

## 运行说明

- 本页 `torchrun` 命令使用 Shell 中的环境设置。日志与训练产物由 `IMAGINAIRE_OUTPUT_ROOT` 指定。
- 重复使用同一输出目录和任务名会自动恢复已有 checkpoint；需要从头测试时，请使用新目录或任务名。
- `checkpoint.save_iter=1000` 只调整周期保存间隔，训练结束时仍会保存最终 checkpoint，请预留存储空间。
- 修改卡数时，同步调整 `HIP_VISIBLE_DEVICES`、`--nproc_per_node` 和 `data_parallel_shard_degree`。
- 调整 Reasoner 训练步数时，同步修改 `trainer.max_iter` 和 `scheduler.cycle_lengths`。
- 单项优化通过模型的环境变量或训练配置控制。
