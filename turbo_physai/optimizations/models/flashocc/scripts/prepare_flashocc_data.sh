#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TURBO_ROOT=$(cd "$MODEL_ROOT/../../../../" && pwd)
ROOT=$(cd "$TURBO_ROOT/.." && pwd)
STATIC_DATA="$ROOT/static_data"
NUSCENES="$STATIC_DATA/data/nuscenes"

if [[ ! -d "$NUSCENES/samples" || ! -d "$NUSCENES/v1.0-trainval" || ! -d "$NUSCENES/gts" ]]; then
    echo "Need samples/, v1.0-trainval/, and gts/ under $NUSCENES" >&2
    exit 2
fi
for name in bevdetv2-nuscenes_infos_train.pkl bevdetv2-nuscenes_infos_val.pkl; do
    if [[ -e "$NUSCENES/$name" ]]; then
        echo "Refusing to overwrite existing metadata: $NUSCENES/$name" >&2
        exit 3
    fi
done

source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv1
export PYTHONPATH="$ROOT/FlashOCC:$ROOT/runtime/mmdetection3d:$ROOT/TurboPhysAI"
cd "$STATIC_DATA"
python "$ROOT/FlashOCC/tools/create_data_bevdet.py"
test -s "$NUSCENES/bevdetv2-nuscenes_infos_train.pkl"
test -s "$NUSCENES/bevdetv2-nuscenes_infos_val.pkl"
echo "FlashOCC train/val metadata created under $NUSCENES"
