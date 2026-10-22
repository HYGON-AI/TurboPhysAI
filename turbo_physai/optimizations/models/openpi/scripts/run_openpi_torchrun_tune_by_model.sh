#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/prepare_tune_runtime.sh"

turbo-physai run \
    --model openpi \
    --log-report \
    -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "pi05_libero_8gpu_bs32_by_model_${STAMP}" \
    --batch-size 256 \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --pytorch_weight_path "${WEIGHT_PATH}" \
    --num_train_steps 10000 \
    --log-interval 1 \
    --save-interval 5000 \
    --no-wandb-enabled \
    2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_model_${STAMP}.log"
