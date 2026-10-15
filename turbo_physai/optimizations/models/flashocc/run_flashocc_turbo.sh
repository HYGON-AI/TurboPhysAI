#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/FlashOCC_TurboPhysAI
FLASHOCC_ROOT="$ROOT/FlashOCC"
STATIC_DATA_ROOT="$ROOT/static_data/data/nuscenes"
work_dir="$ROOT/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)"
cd "$FLASHOCC_ROOT"
exec turbo-physai run \
    --optimization-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml \
    --runtime-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml \
    --log-report \
    -- \
    torchrun --nproc_per_node=8 \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/tools/train.py \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "$work_dir" \
    --cfg-options \
    data.samples_per_gpu=16 \
    data.workers_per_gpu=4 \
    "data.train.data_root=$STATIC_DATA_ROOT/" \
    "data.train.ann_file=$STATIC_DATA_ROOT/bevdetv2-nuscenes_infos_train.pkl" \
    "data.val.data_root=$STATIC_DATA_ROOT/" \
    "data.val.ann_file=$STATIC_DATA_ROOT/bevdetv2-nuscenes_infos_val.pkl" \
    "data.test.data_root=$STATIC_DATA_ROOT/" \
    "data.test.ann_file=$STATIC_DATA_ROOT/bevdetv2-nuscenes_infos_val.pkl" \
    "load_from=$ROOT/static_data/ckpts/bevdet-r50-cbgs.pth" \
    log_config.interval=1
