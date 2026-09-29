# Cosmos3 运行指南

在 `cosmos-framework` 根目录运行训练，通过 `turbo-physai run` 加载优化。以下示例使用 8 张 HCU，训练 70 步。

## 环境准备

### 拉取镜像及创建容器
```
docker pull harbor.sourcefind.cn:5443/hcu/admin/base/custom:cosmos-framework-das-pytorch2.10.0-ubuntu22.04-dtk26.04-py3.10

docker run -it --network=host --name=cosmos_hygon --privileged --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size=512G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -u root --ulimit stack=-1:-1 --ulimit memlock=-1:-1  -v /opt/hyhal:/opt/hyhal:ro  harbor.sourcefind.cn:5443/hcu/admin/base/custom:cosmos-framework-das-pytorch2.10.0-ubuntu22.04-dtk26.04-py3.10
```
### 克隆代码并安装
```bash
git clone https://github.com/HYGON-AI/TurboPhysAI.git
cd TurboPhysAI
pip install -e . --no-build-isolation
cd ..
git clone https://github.com/NVIDIA/cosmos-framework.git
cd cosmos-framework
git checkout 9726697a
```

## 数据与模型权重

按训练任务准备以下资产，路径通过后面的环境变量设置。

| 任务 | 数据 | 模型权重 |
| --- | --- | --- |
| Generator Nano | BridgeData2 的 `sft_dataset_bridge` | Cosmos3-Nano DCP、Wan2.2 VAE、Cosmos3-Nano processor/tokenizer |
| Action Policy | DROID LeRobot 数据集 | Cosmos3-Nano DCP、Wan2.2 VAE、Qwen3-VL-8B tokenizer 缓存 |
| Reasoner | VideoPhy2 或 LLaVA-OV 配方对应数据 | Qwen3-VL 模型目录、Cosmos3-Nano-VLM safetensors |


```bash
cd /data
wget http://42.228.13.241:18000/docker-images/tencent/Cosmos3/BridgeData2-Subset-Synthetic-Captions.zip
unzip BridgeData2-Subset-Synthetic-Captions.zip

wget http://42.228.13.241:18000/docker-images/tencent/Cosmos3/videophysics_official.tar.gz
tar -xzf videophysics_official.tar.gz -C /data/

wget http://42.228.13.241:18000/docker-images/tencent/Cosmos3/Cosmos3-DROID-perf64.tar.gz
tar -xzf Cosmos3-DROID-perf64.tar.gz -C /data/

wget http://42.228.13.241:18000/docker-images/tencent/Cosmos3/Qwen3-VL-8B-Instruct-tokenizer-cache.tar.gz
tar -xzf Qwen3-VL-8B-Instruct-tokenizer-cache.tar.gz -C /data/
```

DROID perf64 子集使用 `/data/Cosmos3-DROID-perf64/keep_ranges_perf64.json` 过滤文件。Tokenizer 缓存解压到 `/data/hf-cache`，不包含 Qwen 模型权重。

将 Cosmos3-Nano HF 权重转换为 DCP 时，在 Cosmos 仓库根目录设置 `WAN_VAE_PATH` 和 `BASE_CHECKPOINT_PATH`，然后执行：

```bash
   WAN_VAE_PATH=$WAN_VAE_PATH HF_HUB_OFFLINE=1 \
   python -m cosmos_framework.scripts.convert_model_to_dcp \
       -i /path/to/Cosmos3-Nano -o $BASE_CHECKPOINT_PATH
   ```

上游转换脚本会读取 VAE/processor；离线环境需预先准备 Hugging Face 缓存。

## 启动训练

所有命令均在 `cosmos-framework` 根目录执行。Generator、Action Policy 和 Reasoner 共用 `--model cosmos3` 默认优化配置，训练场景由 `--sft-toml` 选择。

### Generator Nano

设置数据与权重路径后运行：

```bash
PYTHONPATH=/data/cosmos-test/cosmos-framework
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATASET_PATH=/data/BridgeData2-Subset-Synthetic-Captions/sft_dataset_bridge
export BASE_CHECKPOINT_PATH=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth
export LOCAL_PROCESSOR_DIR=/public/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Nano
export IMAGINAIRE_OUTPUT_ROOT=./outputs/hcu_training

turbo-physai run --model cosmos3 --log-report \
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
PYTHONPATH=/data/cosmos-test/cosmos-framework
export DROID_ROOT=/data/Cosmos/datasets/Cosmos3-DROID-perf64
export KEEP_RANGES_PATH=/data/Cosmos/datasets/Cosmos3-DROID-perf64/keep_ranges_perf64.json
export HF_HOME=/data/cosmos-test/hf-cache
export BASE_CHECKPOINT_PATH=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/Cosmos3-Nano-DCP
export WAN_VAE_PATH=/public/opendas/DL_DATA/llm-models/Cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth
export LOCAL_PROCESSOR_DIR=/public/opendas/DL_DATA/llm-models/Cosmos3/Cosmos3-Nano
export DROID_ROOT=/data/Cosmos/datasets/droid_plus_lerobot_640x360_20260412
export KEEP_RANGES_PATH=/data/Cosmos/datasets/Cosmos3-DROID-perf64/keep_ranges_perf64.json
export IMAGINAIRE_OUTPUT_ROOT=./outputs/action_policy
export TORCHINDUCTOR_WORKER_START=spawn TORCHINDUCTOR_COMPILE_THREADS=4

turbo-physai run --model cosmos3 --log-report --set OMP_NUM_THREADS=4 \
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

本例使用 DROID perf64 子集及对应过滤文件，`MIOPEN_FIND_MODE=1` 由默认 runtime 配置提供。

一次性性能运行如需关闭 checkpoint 写入，追加 `checkpoint.type._target_=cosmos_framework.checkpoint.load_only.LoadOnlyDistributedCheckpointer`；默认保留正常保存行为。

### Reasoner VideoPhy2

```bash
export VIDEOPHYSICS_ROOT=/data/videophysics_official
export VLM_MODEL_NAME=/path/to/Qwen3-VL-8B-Instruct
export VLM_SAFETENSORS_PATH=/path/to/Cosmos3-Nano-VLM
export IMAGINAIRE_OUTPUT_ROOT=./outputs/reasoner

turbo-physai run --model cosmos3 --log-report \
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

沿用上面的 VLM 路径；需要联网时，在 `turbo-physai run` 后添加 `--set HF_HUB_OFFLINE=0`。

```bash
turbo-physai run --model cosmos3 --log-report \
  torchrun --nproc_per_node=8 --master_port=29715 \
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

## 运行说明

- `--model cosmos3` 自动加载 `runtime.yaml`，默认将 `HF_HUB_OFFLINE`、`OMP_NUM_THREADS`、`MIOPEN_FIND_MODE`、`PYTORCH_MIOPEN_SUGGEST_NDHWC`、`TORCH_NCCL_HIGH_PRIORITY` 设为 `1`。使用 `--set NAME=VALUE` 覆盖；配置值优先于 Shell 中的 `export`。
- 修改卡数时，同步调整 `HIP_VISIBLE_DEVICES`、`--nproc_per_node` 和 `data_parallel_shard_degree`。
- 调整 Reasoner 训练步数时，同步修改 `trainer.max_iter` 和 `scheduler.cycle_lengths`。
- `--log-report` 输出各 rank 的优化加载状态；日志与训练产物由 `IMAGINAIRE_OUTPUT_ROOT` 指定。
- 如需关闭某项优化，在 `turbo-physai run` 后添加 `--disable-group <group-id>`，例如 `--disable-group cosmos3.vae_compile`。
- 使用 Cosmos 的 launcher 时，将 Turbo 前缀加在实际 `torchrun` 调用处，并保留注入后的 `PYTHONPATH`。
