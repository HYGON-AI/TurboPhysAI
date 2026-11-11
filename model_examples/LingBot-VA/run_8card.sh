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
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Groups use upstream's top-level modules (train, modules, distributed).
# Expose them before bootstrap while launching from the repository root.
export PYTHONPATH="$LINGBOT_VA_ROOT/wan_va${PYTHONPATH:+:$PYTHONPATH}"
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
cd "$LINGBOT_VA_ROOT"
umask 007
exec turbo-physai run \
  --model lingbot_va \
  --log-report \
  python -m torch.distributed.run \
    --nproc_per_node="${NGPU:-8}" \
    --local-ranks-filter="${LOG_RANK:-0}" \
    --master_port "${MASTER_PORT:-29501}" \
    --tee 3 \
    -m turbo_physai.optimizations.models.lingbot_va.entrypoint \
    --config-name "${CONFIG_NAME:-libero_train}" --save-root "$RUN_ROOT" "$@"
