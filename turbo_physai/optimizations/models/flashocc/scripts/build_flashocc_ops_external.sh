#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
TURBO_ROOT=$(cd "$MODEL_ROOT/../../../../" && pwd)
ROOT=$(cd "$TURBO_ROOT/.." && pwd)
FLASHOCC_ROOT="$ROOT/FlashOCC"
if [[ -n "$(git -C "$FLASHOCC_ROOT" status --porcelain)" ]]; then
    echo "Refusing to build: official FlashOCC checkout is not clean." >&2
    exit 2
fi

for op in bev_pool bev_pool_v2 nearest_assign; do
    if compgen -G "$FLASHOCC_ROOT/projects/mmdet3d_plugin/ops/$op/${op}_ext*.so" >/dev/null; then
        echo "Refusing to overwrite existing $op extension." >&2
        exit 2
    fi
done
BUILD_SRC=""
BUILD_TMP=""
cleanup_build_dirs() {
    if [[ -n "$BUILD_SRC" ]]; then
        rm -rf -- "$BUILD_SRC"
    fi
    if [[ -n "$BUILD_TMP" ]]; then
        rm -rf -- "$BUILD_TMP"
    fi
}
trap cleanup_build_dirs EXIT

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
    --build-lib "$FLASHOCC_ROOT/projects" \
    --build-temp "$BUILD_TMP"

for extension in \
    "$FLASHOCC_ROOT/projects/mmdet3d_plugin/ops/bev_pool"/bev_pool_ext*.so \
    "$FLASHOCC_ROOT/projects/mmdet3d_plugin/ops/bev_pool_v2"/bev_pool_v2_ext*.so \
    "$FLASHOCC_ROOT/projects/mmdet3d_plugin/ops/nearest_assign"/nearest_assign_ext*.so; do
    if [[ ! -f "$extension" ]]; then
        echo "Expected external HCU extension missing: $extension" >&2
        exit 3
    fi
done

export FLASHOCC_ROOT
cd "$FLASHOCC_ROOT"
TURBO_PHYSAI_BOOTSTRAP=1 python - <<'PY'
import importlib
import turbo_physai.optimizations.models.flashocc

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
