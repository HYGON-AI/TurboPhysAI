#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TURBO_ROOT=$(cd "$MODEL_ROOT/../../../../" && pwd)
ROOT=$(cd "$TURBO_ROOT/.." && pwd)
STATIC_DATA="$ROOT/static_data"
NUSCENES="$STATIC_DATA/data/nuscenes"
NUSCENES_ARCHIVES="$STATIC_DATA/downloads/nuscenes"
OCC_DOWNLOAD="$STATIC_DATA/downloads/occupancy"
OCC_REPO=OpenDriveLab/CVPR2023-3D-Occupancy
MIN_FREE_GB=350

mkdir -p "$NUSCENES" "$NUSCENES_ARCHIVES" "$OCC_DOWNLOAD" "$STATIC_DATA/ckpts"
available_gb=$(df -BG --output=avail "$STATIC_DATA" | tail -n 1 | tr -cd '0-9')
if [[ -z "$available_gb" ]]; then
    available_gb=0
fi
if [[ "$available_gb" -lt "$MIN_FREE_GB" ]]; then
    echo "static_data has $available_gb GB free; need at least $MIN_FREE_GB GB." >&2
    echo "Bind-mount a writable host directory at $STATIC_DATA before downloading." >&2
    echo "Keeping compressed archives and extracted files together needs additional space." >&2
    exit 2
fi

echo "Download NuScenes v1.0 trainval from https://www.nuscenes.org/download"
echo "Place its .tgz/.tar.gz files under $NUSCENES_ARCHIVES"
echo "This script never removes downloaded archives."
for archive in "$NUSCENES_ARCHIVES"/*.tgz "$NUSCENES_ARCHIVES"/*.tar.gz; do
    [[ -f "$archive" ]] || continue
    echo "Extracting $archive"
    tar -xzf "$archive" -C "$NUSCENES"
done

if [[ ! -d "$NUSCENES/gts" ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate mmcv1
    if ! command -v openxlab >/dev/null; then
        python -m pip install openxlab
    fi
    if [[ ! -f "$HOME/.openxlab/config.json" ]]; then
        openxlab login
    fi
    openxlab dataset get --dataset-repo "$OCC_REPO" --target-path "$OCC_DOWNLOAD"
    gts_source=$(find "$OCC_DOWNLOAD" -type d -path '*/trainval/gts' -print -quit)
    if [[ -z "$gts_source" ]]; then
        echo "trainval/gts was not found under $OCC_DOWNLOAD; inspect the OpenXLab result." >&2
        exit 3
    fi
    ln -s "$gts_source" "$NUSCENES/gts"
fi

echo "Occupancy data is in place if the gts symlink/directory exists."
echo "Place bevdet-r50-cbgs.pth manually at $STATIC_DATA/ckpts/"
echo "Official BEVDet model zoo Baidu share: https://pan.baidu.com/s/1237QyV18zvRJ1pU3YzRItw (code npe1)"
echo "Then run $MODEL_ROOT/scripts/prepare_flashocc_data.sh."
