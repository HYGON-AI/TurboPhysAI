#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

cd /workspace/FlashOCC_TurboPhysAI/FlashOCC
turbo-physai run \
    --optimization-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml \
    --runtime-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml \
    --log-report \
    -- \
    torchrun --nproc_per_node=2 --master_port=29500 \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/tools/train.py \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "/workspace/FlashOCC_TurboPhysAI/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)" \
    --cfg-options \
    data.samples_per_gpu=24 \
    data.workers_per_gpu=24 \
    evaluation.start=24 \
    data.train.data_root=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/ \
    data.train.ann_file=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl \
    data.val.data_root=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/ \
    data.val.ann_file=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl \
    data.test.data_root=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/ \
    data.test.ann_file=/workspace/FlashOCC_TurboPhysAI/static_data/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl \
    load_from=/workspace/FlashOCC_TurboPhysAI/static_data/ckpts/bevdet-r50-cbgs.pth \
    log_config.interval=1
