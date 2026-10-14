#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
STATIC_DATA="$ROOT/static_data"
WORK_DIR="$ROOT/work_dirs/flashocc_turbo_$(date +%Y%m%d_%H%M%S)"

if [[ -f /root/.hygon_base_env ]]; then
    source /root/.hygon_base_env
fi
source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1

for path in \
    "$STATIC_DATA/data/nuscenes/samples" \
    "$STATIC_DATA/data/nuscenes/v1.0-trainval" \
    "$STATIC_DATA/data/nuscenes/gts" \
    "$STATIC_DATA/data/nuscenes/bevdetv2-nuscenes_infos_train.pkl" \
    "$STATIC_DATA/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl" \
    "$STATIC_DATA/ckpts/bevdet-r50-cbgs.pth"; do
    if [[ ! -e "$path" ]]; then
        printf '[FlashOCC] Required asset missing: %s\n' "$path" >&2
        exit 2
    fi
done

mkdir -p "$WORK_DIR"
cd "$STATIC_DATA"
exec turbo-physai run \
    --optimization-config "$SCRIPT_DIR/configs/optimization.yaml" \
    --runtime-config "$SCRIPT_DIR/configs/runtime.yaml" \
    --log-report \
    -- \
    torchrun \
    --nproc_per_node=8 \
    --master_addr=localhost \
    --master_port=6000 \
    "$ROOT/FlashOCC/tools/train.py" \
    "$ROOT/FlashOCC/projects/configs/flashocc/flashocc-r50.py" \
    --launcher pytorch \
    --work-dir "$WORK_DIR" \
    --cfg-options data.samples_per_gpu=8 data.workers_per_gpu=4
