#!/usr/bin/env bash
set -euo pipefail

source /root/.hygon_base_env
cd /workspace/FlashOCC

work_dir="work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)"
exec turbo-physai run \
    --optimization-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/optimization.yaml \
    --runtime-config /workspace/FlashOCC_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/flashocc/configs/runtime.yaml \
    --log-report \
    -- \
    torchrun --nproc_per_node=4 \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/tools/train.py \
    /workspace/FlashOCC_TurboPhysAI/FlashOCC/projects/configs/flashocc/flashocc-r50.py \
    --launcher pytorch \
    --work-dir "$work_dir" \
    --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4
