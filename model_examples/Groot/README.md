# GR00T 应用说明

## 1. 模型简介

[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) 是 NVIDIA 的视觉-语言-动作（VLA）模型，使用 Cosmos Reason2 / Qwen3-VL 骨干处理图像和语言，通过动作 DiT 生成机器人动作。本文使用 GR00T N1.7 3B 权重，在 LIBERO 10 数据集上进行微调，训练入口为 `gr00t/experiment/launch_finetune.py`。

## 2. 优化接入基线

TurboPhysAI 的 GR00T 优化以 Isaac-GR00T 官方提交 `51d4c89f72fda44cbf77285c6a8114b52676b8a1` 为基线。本文使用 OpenCV 在 CPU 上解码视频。 [训练配置与启动步骤](#7-通过-physai-启动训练)

## 3. 构建训练环境

### 3.1 创建容器

按照[安装指南](../../docs/zh/get_started/installation.md#使用配套镜像)，使用 VLA 配套镜像创建容器。下文命令均在容器内执行。

### 3.2 准备 TurboPhysAI 源码和 GR00T 训练依赖

将 `PHYS_ROOT` 设置为容器内包含 GR00T 优化子包的 TurboPhysAI 源码目录。编译环境要求见[安装指南](../../docs/zh/get_started/installation.md#从源码安装)。

```bash
export PHYS_ROOT=/workspace/TurboPhysAI
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

镜像已提供 DTK 版 PyTorch、DeepSpeed 和 Apex 等基础包。GR00T 训练还需要安装 [requirements.txt](requirements.txt) 中的额外依赖，安装前先锁定镜像中的厂商包版本，避免 pip 替换这些组件。

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
# 确认安装计划不会替换上述厂商加速包后执行
python3 -m pip install \
  -r "$PHYS_ROOT/model_examples/Groot/requirements.txt" \
  -c "$GROOT_CONFIG_DIR/vendor-constraints.txt"

# 在源码目录编译扩展，并让训练进程使用同一份源码
cd "$PHYS_ROOT"
MAX_JOBS=8 python3 setup.py build_ext --inplace
export PYTHONPATH="$PHYS_ROOT:$GROOT_MODEL${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch, turbo_physai._C'
```

## 4. 准备模型源码

准备源码，克隆 Isaac-GR00T 官方仓库并切换到基线提交。

```bash
mkdir -p "$(dirname "$GROOT_MODEL")"
git clone https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_MODEL"
git -C "$GROOT_MODEL" switch --detach 51d4c89f72fda44cbf77285c6a8114b52676b8a1
```

## 5. 准备 LIBERO 10 数据

原始H.264资源通过 `$DATASET_SOURCE` 读取，训练统一使用 `$DATASET_DIR` 的可写元数据视图。已有资源时跳过下面的下载、转码和编码检查，但仍需执行末尾的“创建训练视图”。从头准备数据时，转码和编码检查需要 `ffmpeg` 和 `ffprobe`。

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

已有 GR00T 和 Cosmos 权重时，可直接设置下方的本地权重路径。需要下载时，先在 [Cosmos 模型页](https://huggingface.co/nvidia/Cosmos-Reason2-2B)获得访问资格，再执行：

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

复制内置 Recipe，生成启用全部 Group 的优化配置。

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

使用全局 batch 32 和 BF16 运行 10 步，检查训练启动和末步保存。该示例仅保存模型权重。此前 HCU 数值对照覆盖 100 步。

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

日志应显示 18 个 Group 为 `applied`、训练达到 `10/10` 并输出 `Training completed!`。

### 7.3 单机八卡训练

NVIDIA 的 LIBERO 10（Long）复现配方使用全局 batch 640、20,000 步。 [官方训练命令](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/LIBERO/README.md#L39)

使用第 7.1 节生成的配置，每 1,000 步保存完整 checkpoint。

```bash
set -euo pipefail
export NUM_GPUS=8 GLOBAL_BATCH_SIZE=640 MAX_STEPS=20000 SAVE_STEPS=1000
unset GR00T_DIT_SDPA_MODE

RUN_DIR="$DATA_ROOT/outputs/groot-libero10-$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "$RUN_DIR"
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
    --output-dir "$RUN_DIR/checkpoints" --state-dropout-prob 0.2 \
  2>&1 | tee "$RUN_DIR/train.log"
```

续训时沿用相同的卡数、优化配置和 `RUN_DIR/checkpoints` 输出目录，在上述 `examples/finetune.sh` 命令末尾增加 `--resume-from-checkpoint`。使用原训练的 `RUN_DIR`，并将 `MAX_STEPS` 设置为续训的目标总步数。
