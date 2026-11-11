#!/usr/bin/env bash
set -euo pipefail

: "${LINGBOT_VA_ROOT:?Set LINGBOT_VA_ROOT to the pinned LingBot-VA checkout}"
: "${RUN_ROOT:?Set RUN_ROOT to a new output directory}"

LINGBOT_VA_ROOT="$(cd "$LINGBOT_VA_ROOT" && pwd)"
if [[ ! -f "$LINGBOT_VA_ROOT/wan_va/train.py" || ! -f "$LINGBOT_VA_ROOT/wan_va/modules/model.py" ]]; then
  echo "LINGBOT_VA_ROOT must point to the LingBot-VA repository root" >&2
  exit 2
fi

CACHE_ROOT="${CACHE_ROOT:-$RUN_ROOT/cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LD_LIBRARY_PATH="/opt/dtk/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}"
unset ROCBLAS_TENSILE_LIBPATH TORCHINDUCTOR_MAX_AUTOTUNE TORCHINDUCTOR_COORDINATE_DESCENT_TUNING
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

cd "$LINGBOT_VA_ROOT/wan_va"
exec turbo-physai run \
  --model lingbot_va \
  --log-report \
  torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    -m turbo_physai.optimizations.models.lingbot_va.entrypoint \
    --config-name libero_train --save-root "$RUN_ROOT"
