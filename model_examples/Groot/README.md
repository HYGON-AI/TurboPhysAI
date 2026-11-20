# GR00T 应用说明

## 1. 模型简介

[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) 是 NVIDIA 的视觉-语言-动作（VLA）模型，使用 Cosmos Reason2 / Qwen3-VL 骨干处理图像和语言，通过动作 DiT 生成机器人动作。本文使用 GR00T N1.7 3B 权重，在 LIBERO 10 数据集上进行微调，训练入口为 `gr00t/experiment/launch_finetune.py`。

## 2. 优化接入基线

TurboPhysAI 基于官方提交 `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 接入，保持模型源码不变。本文使用 OpenCV CPU 解码。下表列出示例所需的兼容 Group，完整配置见[第 7 节](#7-通过-physai-启动训练)。

| Group | 作用 |
| --- | --- |
| `groot.compat.training_config` | 检查 TF32 支持情况，并通过 `VLM_MODEL_PATH` 使用本地 Cosmos。 |
| `groot.compat.opencv_video` | 将训练的数据索引帧读取切换为 OpenCV。 |
| `groot.dit.hip_math_sdpa` | HIP 环境下为动作 DiT 使用 Math SDPA，规避已观察到的数值异常。 |

## 3. 构建训练环境

### 3.1 创建容器

容器创建方式参见[安装指南](../../docs/zh/get_started/installation.md#创建容器)。在宿主机执行以下命令，将 `HOST_WORKSPACE` 替换为实际工作目录，该目录挂载为容器内的 `/workspace`：

```bash
set -euo pipefail
image=harbor.sourcefind.cn:5443/hcu/admin/base/custom:dit-vla-wm-base-ubuntu22.04-dtk26.04-torch2.10-py3.10
export HOST_WORKSPACE=/path/to/groot-workspace
mkdir -p "$HOST_WORKSPACE"

docker pull "$image"
docker run -dit \
  --network=host \
  --name=physai_groot \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --shm-size=128G \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  --workdir=/workspace \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_WORKSPACE":/workspace \
  --entrypoint /bin/bash \
  "$image"

docker exec -it physai_groot bash
```

**复用本地数据和权重：** 在 `docker run` 命令的 `--entrypoint` 之前增加对应只读挂载参数；数据源在容器内使用 `_h264_source` 路径，可写训练视图另建于工作区。源码按后文准备，无需单独挂载。

| 资源 | 宿主机位置 |
| --- | --- |
| LIBERO H.264 训练数据 | `/public/home/zhaobx/Groot/data/libero_10_no_noops_1.0.0_lerobot_h264` |
| GR00T 起始权重 | `/public/home/zhaobx/Groot/data/checkpoints/GR00T-N1.7-3B-local-cosmos` |
| Cosmos 骨干权重 | `/public/home/zhaobx/Groot/data/checkpoints/nvidia/Cosmos-Reason2-2B` |

```text
  --mount type=bind,src=/public/home/zhaobx/Groot/data/libero_10_no_noops_1.0.0_lerobot_h264,dst=/workspace/data/libero_10_no_noops_1.0.0_lerobot_h264_source,readonly \
  --mount type=bind,src=/public/home/zhaobx/Groot/data/checkpoints/GR00T-N1.7-3B-local-cosmos,dst=/workspace/data/checkpoints/GR00T-N1.7-3B-local-cosmos,readonly \
  --mount type=bind,src=/public/home/zhaobx/Groot/data/checkpoints/nvidia/Cosmos-Reason2-2B,dst=/workspace/data/checkpoints/nvidia/Cosmos-Reason2-2B,readonly \
```

源目录需已存在，已有容器需重新创建才能增加挂载。下文均在容器内使用 Bash 执行。检查八卡设备并安装系统工具：

```bash
set -euo pipefail
hy-smi
python3 - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.device_count())
assert torch.version.hip and torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
PY

apt-get update
apt-get install -y git git-lfs ffmpeg
git lfs install
```

### 3.2 准备 TurboPhysAI 与训练依赖

将包含 GR00T 优化子包的源码放到 `/workspace/TurboPhysA`，在源码目录编译原生扩展，后续直接从源码导入 TurboPhysAI。编译环境要求见[安装说明](../../docs/zh/get_started/installation.md#从源码安装)。

```bash
export PHYS_ROOT=/workspace/TurboPhysA
export GROOT_OPT="$PHYS_ROOT/turbo_physai/optimizations/models/groot"
export GROOT_MODEL="$PHYS_ROOT/model/Isaac-GR00T-official"
export DATA_ROOT=/workspace/data
export DATASET_SOURCE="$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_h264_source"
export DATASET_DIR="$DATA_ROOT/views/libero10-h264"
export BASE_MODEL_SOURCE="$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos"
export HF_HOME="$DATA_ROOT/cache/huggingface"
export PYTHONUNBUFFERED=1
export GROOT_CONFIG_DIR="$DATA_ROOT/configs/groot-opencv-baseline"
test -f "$GROOT_OPT/configs/recipe.yaml"
mkdir -p "$GROOT_CONFIG_DIR" "$DATA_ROOT/outputs" "$DATA_ROOT/logs"
```

示例需要镜像中的 DTK 加速栈，包括厂商版 DeepSpeed 和 Apex。额外依赖见 [requirements.txt](requirements.txt)。先锁定厂商包版本再安装，避免替换加速组件，不使用官方 `uv sync`，其 Python/CUDA 依赖与本文 DTK 环境不同。

```bash
python3 - <<'PY' > "$GROOT_CONFIG_DIR/vendor-constraints.txt"
from importlib.metadata import version, PackageNotFoundError
for name in ("torch", "torchvision", "torchaudio", "deepspeed", "apex", "flash-attn", "triton", "torchcodec"):
    try:
        print(f"{name}=={version(name)}")
    except PackageNotFoundError:
        if name not in ("torchaudio", "torchcodec"):
            raise
PY

python3 -m pip install --dry-run \
  -r "$PHYS_ROOT/model_examples/Groot/requirements.txt" \
  -c "$GROOT_CONFIG_DIR/vendor-constraints.txt"
# 确认安装计划不会替换上述厂商加速包后执行。
python3 -m pip install \
  -r "$PHYS_ROOT/model_examples/Groot/requirements.txt" \
  -c "$GROOT_CONFIG_DIR/vendor-constraints.txt"

# 在源码目录编译扩展，并让训练进程使用同一份源码。
cd "$PHYS_ROOT"
MAX_JOBS=8 python3 setup.py build_ext --inplace
export PYTHONPATH="$PHYS_ROOT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'CHECK_IMPORT'
import torch
import turbo_physai
from turbo_physai import _C
print("框架加载位置：", turbo_physai.__file__)
print("原生扩展位置：", _C.__file__)
CHECK_IMPORT
python3 -m turbo_physai.cli --help
```

## 4. 准备模型源码

在新机器获取官方仓库并锁定基线：

```bash
mkdir -p "$(dirname "$GROOT_MODEL")"
git clone https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_MODEL"
git -C "$GROOT_MODEL" switch --detach 51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

已有官方源码时，跳过 clone 和切换提交，确认 `git -C "$GROOT_MODEL" rev-parse HEAD` 为上述提交且工作区干净。若出现 `dubious ownership`，执行：

```bash
git config --global --add safe.directory "$GROOT_MODEL"
```

## 5. 准备 LIBERO 10 数据

原始H.264资源通过 `$DATASET_SOURCE` 读取，训练统一使用 `$DATASET_DIR` 的可写元数据视图。已有资源时跳过下面的下载、转码和编码检查，但仍需执行末尾的“创建训练视图”。没有本地资源时下载并转码：

```bash
export DATASET_ORIGINAL="$DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_original"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  --repo-type dataset --revision e1a223d30b896c1613f270a2bfc63d382b3de7e1 \
  --local-dir "$DATASET_ORIGINAL"

# 保留原始AV1副本，转码只在新目录操作。
test ! -e "$DATASET_SOURCE"
cp -a "$DATASET_ORIGINAL" "$DATASET_SOURCE"
python3 "$GROOT_MODEL/examples/SimplerEnv/convert_av1_to_h264.py" \
  "$DATASET_SOURCE/videos" -j 4
cp "$GROOT_MODEL/examples/LIBERO/modality.json" "$DATASET_SOURCE/meta/modality.json"
```

核对转码结果并更新编码元数据：

```bash
python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
source, target = Path(os.environ["DATASET_ORIGINAL"]), Path(os.environ["DATASET_SOURCE"])
def probe(path):
    args = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,nb_read_frames,width,height,avg_frame_rate",
            "-of", "json", str(path)]
    return json.loads(subprocess.check_output(args))["streams"][0]
videos = list(source.glob("videos/**/*.mp4"))
assert videos, "没有找到源视频"
for original in videos:
    converted = target / original.relative_to(source)
    before, after = probe(original), probe(converted)
    assert after["codec_name"] == "h264", converted
    for key in ("nb_read_frames", "width", "height", "avg_frame_rate"):
        assert before[key] == after[key], (converted, key, before[key], after[key])
info_path = target / "meta/info.json"
info = json.loads(info_path.read_text())
for feature in info["features"].values():
    if feature.get("dtype") == "video":
        feature.setdefault("info", {})["video.codec"] = "h264"
info_path.write_text(json.dumps(info, indent=2) + "\n")
print(f"已核对{len(videos)}个视频并更新编码元数据")
PY
```

**创建训练视图：** 复制可写的 `meta/`，将 `data/`、`videos/` 链接到容器内可见的原始资源。已有同一来源的视图保留其统计文件；不同实验可另设 `DATASET_DIR`。

```bash
python3 - <<'DATA_VIEW'
import os, shutil
from pathlib import Path
source = Path(os.environ["DATASET_SOURCE"]).resolve(strict=True)
view = Path(os.environ["DATASET_DIR"]).absolute()
assert source != view.resolve(), "训练视图不能覆盖原始数据"
for name in ("meta", "data", "videos"):
    assert (source / name).is_dir(), source / name
view.mkdir(parents=True, exist_ok=True)
meta = view / "meta"
assert not meta.is_symlink(), "meta必须是可写副本，不能链接到原始数据"
if not meta.exists():
    shutil.copytree(source / "meta", meta, copy_function=shutil.copyfile)
meta.chmod(meta.stat().st_mode | 0o700)
for name in ("data", "videos"):
    link = view / name
    if link.is_symlink():
        assert link.resolve() == source / name, "视图已指向其他数据，请更换DATASET_DIR"
    else:
        assert not link.exists(), link
        link.symlink_to(source / name, target_is_directory=True)
assert (meta / "info.json").is_file() and (meta / "modality.json").is_file()
print("训练数据视图：", view)
DATA_VIEW
```

官方训练初始化会生成或更新 `meta/stats.json`、`meta/relative_stats.json`，相对统计即使缓存命中也可能写回，因此不能仅凭已有统计文件将整个训练目录只读使用。下面所有训练命令均传入 `$DATASET_DIR`，不直接使用只读源目录。

## 6. 准备模型权重

权重位置见第3.1节。已有权重时跳过下载，直接设置本地加载配置；否则先在 [Cosmos 模型页](https://huggingface.co/nvidia/Cosmos-Reason2-2B)获得访问资格，再执行：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf auth login
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 \
  --local-dir "$DATA_ROOT/checkpoints/GR00T-N1.7-3B-local-cosmos"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 hf download \
  nvidia/Cosmos-Reason2-2B --revision 9ce19a195e423419c349abfc86fd07178b230561 \
  --local-dir "$DATA_ROOT/checkpoints/nvidia/Cosmos-Reason2-2B"
```

设置本地权重路径：

```bash
export BASE_MODEL_PATH="$BASE_MODEL_SOURCE"
export VLM_MODEL_PATH="$DATA_ROOT/checkpoints/nvidia/Cosmos-Reason2-2B"
```

确认 GR00T 权重目录的 `config.json` 中 `model_name` 指向本地 Cosmos 路径。路径需包含 `nvidia/Cosmos-Reason2`，供官方代码识别骨干类型。新下载的权重如不匹配，需先在可写副本中调整该字段，再将 `BASE_MODEL_PATH` 指向该副本。训练时 `VLM_MODEL_PATH` 也会传给处理器。训练前开启离线模式：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_WANDB=0
```

## 7. 通过 PhysAI 启动训练

### 7.1 生成优化配置

复制仓库 Recipe 生成独立配置，开启全部兼容项和优化项，不修改仓库原配置：

```bash
python3 - <<'PY'
import os
from pathlib import Path
import yaml
source = Path(os.environ["GROOT_OPT"]) / "configs/recipe.yaml"
target = Path(os.environ["GROOT_CONFIG_DIR"])
target.mkdir(parents=True, exist_ok=True)
recipe = yaml.safe_load(source.read_text())
for group in recipe["optimization_groups"]:
    group["enabled"] = True
recipe["metadata"]["id"] = "model.groot.opencv-all.hcu"
(target / "recipe.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
PY

cd "$GROOT_MODEL"
python3 -m turbo_physai.cli optimization generate \
  --recipe "$GROOT_CONFIG_DIR/recipe.yaml" \
  --repo "$GROOT_MODEL" \
  --commit 51d4c89f72fda44cbf77285c6a8114b52676b8a1 \
  --output "$GROOT_CONFIG_DIR/optimization.yaml" --force
python3 -m turbo_physai.cli optimization check \
  "$GROOT_CONFIG_DIR/optimization.yaml" --repo "$GROOT_MODEL"
```

### 7.2 快速验证（单机八卡）

运行10步，全局batch32、每卡4，BF16混合精度。使用官方入口，末步会保存checkpoint并另存最终模型，10步仅验证训练链路，不作为稳定性能或精度结论。

```bash
set -o pipefail
export NUM_GPUS=8 GLOBAL_BATCH_SIZE=32 MAX_STEPS=10
export SAVE_STEPS=1000 DATALOADER_NUM_WORKERS=0 USE_WANDB=0
export MASTER_PORT=29517
unset GR00T_DIT_SDPA_MODE
OUTPUT_DIR="$DATA_ROOT/outputs/groot-opencv-smoke-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUTPUT_DIR"
cd "$GROOT_MODEL"
python3 -m turbo_physai.cli run --disable-numa --log-report \
  --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
  --optimization-config "$GROOT_CONFIG_DIR/optimization.yaml" \
  --set "VLM_MODEL_PATH=$VLM_MODEL_PATH" \
  --set VIDEO_DECODER_BACKEND=opencv -- \
  bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_DIR" \
    --embodiment-tag LIBERO_PANDA --experiment-name libero10_finetune \
    --output-dir "$OUTPUT_DIR" --state-dropout-prob 0.2 --save-only-model \
  2>&1 | tee "$OUTPUT_DIR/train.log"
```

退出码应为0，日志显示全部 19 项 Group 为 `applied`，训练达到 `10/10` 并输出 `Training completed!`。

### 7.3 单机八卡训练

正式训练使用官方入口，保留 checkpoint 与最终模型。AllGather pipeline 不支持训练期间保存 checkpoint，因此先关闭该项并重新生成配置：

```bash
python3 - <<'PY'
import os
from pathlib import Path
import yaml
path = Path(os.environ["GROOT_CONFIG_DIR"]) / "recipe.yaml"
recipe = yaml.safe_load(path.read_text())
for group in recipe["optimization_groups"]:
    if group["id"] == "groot.zero2.allgather_pipeline":
        group["enabled"] = False
path.write_text(yaml.safe_dump(recipe, sort_keys=False))
PY
python3 -m turbo_physai.cli optimization generate \
  --recipe "$GROOT_CONFIG_DIR/recipe.yaml" --repo "$GROOT_MODEL" \
  --commit 51d4c89f72fda44cbf77285c6a8114b52676b8a1 \
  --output "$GROOT_CONFIG_DIR/optimization.yaml" --force
```

启动训练：

```bash
set -o pipefail
export NUM_GPUS=8 GLOBAL_BATCH_SIZE=32 MAX_STEPS=10000
export SAVE_STEPS=1000 DATALOADER_NUM_WORKERS=4 USE_WANDB=0
export MASTER_PORT=29517
unset GR00T_DIT_SDPA_MODE
OUTPUT_DIR="$DATA_ROOT/outputs/groot-opencv-train-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$OUTPUT_DIR"
cd "$GROOT_MODEL"
python3 -m turbo_physai.cli run --disable-numa --log-report \
  --runtime-config "$GROOT_OPT/configs/runtime.yaml" \
  --optimization-config "$GROOT_CONFIG_DIR/optimization.yaml" \
  --set "VLM_MODEL_PATH=$VLM_MODEL_PATH" \
  --set VIDEO_DECODER_BACKEND=opencv -- \
  bash examples/finetune.sh \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_DIR" \
    --embodiment-tag LIBERO_PANDA --experiment-name libero10_finetune \
    --output-dir "$OUTPUT_DIR" --state-dropout-prob 0.2 --save-only-model \
  2>&1 | tee "$OUTPUT_DIR/train.log"
```

训练参数为学习率 `1e-4`、weight decay `1e-5`、warmup `0.05`、梯度累积1；每1000步保存，最多保留5份checkpoint，结束时另存最终模型。`--save-only-model` 不保存优化器、调度器和RNG状态，需要精确续训时去掉该参数。
