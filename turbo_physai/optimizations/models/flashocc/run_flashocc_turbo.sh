#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1

ROOT=/workspace/FlashOCC_TurboPhysAI
STATIC_DATA="$ROOT/static_data"
WORK_DIR="$ROOT/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)"
cd "$STATIC_DATA"

exec turbo-physai run \
    --optimization-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml \
    --runtime-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml \
    --log-report \
    -- \
    torchrun --nproc_per_node=4 \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/tools/train.py \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "$WORK_DIR" \
    --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4 log_config.interval=1
