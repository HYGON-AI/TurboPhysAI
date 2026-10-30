#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TURBO_ROOT=$(cd "$MODEL_ROOT/../../../../" && pwd)
ROOT=$(cd "$TURBO_ROOT/.." && pwd)
FLASHOCC_ROOT="$ROOT/FlashOCC"
RUNTIME_ROOT="$ROOT/runtime"

if [[ -f /root/.hygon_base_env ]]; then
    source /root/.hygon_base_env
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[FlashOCC] Using active Python environment: $CONDA_PREFIX"
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate mmcv1
else
    echo "[FlashOCC] Conda not found; using $(command -v python)" >&2
fi

if [[ -n "$(git -C "$FLASHOCC_ROOT" status --porcelain)" ]]; then
    echo "Refusing to build: official FlashOCC checkout is not clean." >&2
    exit 2
fi

mkdir -p "$RUNTIME_ROOT"
BUILD_SRC=$(mktemp -d /tmp/flashocc-ops-src.XXXXXX)
BUILD_TMP=$(mktemp -d /tmp/flashocc-ops-build.XXXXXX)
PROJECTS="$FLASHOCC_ROOT/projects"
cp "$PROJECTS/setup.py" "$BUILD_SRC/setup.py"
mkdir -p "$BUILD_SRC/mmdet3d_plugin/ops"

for op in bev_pool bev_pool_v2 nearest_assign; do
    mkdir -p "$BUILD_SRC/mmdet3d_plugin/ops/$op"
    cp -a "$PROJECTS/mmdet3d_plugin/ops/$op/src" "$BUILD_SRC/mmdet3d_plugin/ops/$op/src"
done

cd "$BUILD_SRC"
FORCE_CUDA=1 MAX_JOBS=8 python setup.py build_ext \
    --build-lib "$RUNTIME_ROOT/projects" \
    --build-temp "$BUILD_TMP"

for extension in \
    "$RUNTIME_ROOT/projects/mmdet3d_plugin/ops/bev_pool"/bev_pool_ext*.so \
    "$RUNTIME_ROOT/projects/mmdet3d_plugin/ops/bev_pool_v2"/bev_pool_v2_ext*.so \
    "$RUNTIME_ROOT/projects/mmdet3d_plugin/ops/nearest_assign"/nearest_assign_ext*.so; do
    if [[ ! -f "$extension" ]]; then
        echo "Expected external HCU extension missing: $extension" >&2
        exit 3
    fi
done

export FLASHOCC_ROOT
export FLASHOCC_RUNTIME_ROOT="$RUNTIME_ROOT"
export PYTHONPATH="$MODEL_ROOT:$RUNTIME_ROOT/mmdetection3d:$FLASHOCC_ROOT"
python - <<'PY'
import importlib
import os
from flashocc_optimization.compatibility import install_compatibility

install_compatibility(os.environ["FLASHOCC_ROOT"], os.environ["FLASHOCC_RUNTIME_ROOT"])
for name in (
    "projects.mmdet3d_plugin.ops.bev_pool.bev_pool_ext",
    "projects.mmdet3d_plugin.ops.bev_pool_v2.bev_pool_v2_ext",
    "projects.mmdet3d_plugin.ops.nearest_assign.nearest_assign_ext",
):
    module = importlib.import_module(name)
    print(f"{name}: {module.__file__}")
PY

if [[ -n "$(git -C "$FLASHOCC_ROOT" status --porcelain)" ]]; then
    echo "Official FlashOCC changed during external build; inspect immediately." >&2
    exit 4
fi
