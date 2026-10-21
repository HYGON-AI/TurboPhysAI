#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/prepare_tune_runtime.sh"

export PYTHONHASHSEED=0
export OPENPI_SEED=42

turbo-physai run \
    --optimization-config "${OPENPI_MODEL_DIR}/configs/optimization.yaml" \
    --runtime-config "${OPENPI_MODEL_DIR}/configs/runtime.yaml" \
    --log-report \
    -- \
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi05_libero \
    --exp_name "pi05_libero_8gpu_bs32_by_config_${STAMP}" \
    --batch-size 256 \
    --assets-base-dir "${ASSETS_BASE_DIR}" \
    --checkpoint-base-dir "${CHECKPOINT_BASE_DIR}" \
    --pytorch_weight_path "${WEIGHT_PATH}" \
    --num_train_steps 10000 \
    --log-interval 1 \
    --save-interval 1000 \
    --no-wandb-enabled \
    --seed 42 \
    2>&1 | tee -a "${LOG_DIR}/pi05_libero_8gpu_bs32_by_config_${STAMP}.log"
