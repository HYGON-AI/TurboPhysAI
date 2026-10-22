#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

set -Eeuo pipefail

# Resolve user-provided relative paths against the directory where the launcher
# was started, before this helper changes into the OpenPI checkout.
OPENPI_LAUNCH_CWD="$(pwd -P)"
openpi_resolve_path() {
    local path="$1"
    if [[ "${path}" == /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${OPENPI_LAUNCH_CWD}" "${path}"
    fi
}

# Source this file from the model launcher; repository paths are derived from this package.
OPENPI_MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TURBO_ROOT="$(cd "${OPENPI_MODEL_DIR}/../../../.." && pwd -P)"
WORKSPACE_ROOT="$(cd "${TURBO_ROOT}/.." && pwd -P)"
MODEL_REPO="$(cd "$(openpi_resolve_path "${OPENPI_MODEL_REPO:-${WORKSPACE_ROOT}/openpi}")" && pwd -P)"
WEIGHT_PATH="$(openpi_resolve_path "${OPENPI_WEIGHT_PATH:-${WORKSPACE_ROOT}/static_data/pi05_base_pytorch}")"
ASSETS_BASE_DIR="${OPENPI_ASSETS_BASE_DIR:-${WORKSPACE_ROOT}/static_data/assets}"
export OPENPI_ASSETS_DIR="${OPENPI_ASSETS_DIR:-${ASSETS_BASE_DIR}}"
OUTPUT_ROOT="$(openpi_resolve_path "${OPENPI_OUTPUT_ROOT:-${WORKSPACE_ROOT}/output}")"
CHECKPOINT_BASE_DIR="$(openpi_resolve_path "${OPENPI_CHECKPOINT_BASE_DIR:-${OUTPUT_ROOT}/checkpoints}")"
LOG_DIR="$(openpi_resolve_path "${OPENPI_LOG_DIR:-${OUTPUT_ROOT}/logs}")"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUNTIME_ROOT="$(openpi_resolve_path "${OPENPI_RUNTIME_ROOT:-/dev/shm/openpi_8gpu_runtime_${STAMP}_$$}")"
# RuntimeConfig owns stable runtime variables; keep only per-run shell scratch here.
export TMPDIR="${RUNTIME_ROOT}/tmp"
export PYTHONPYCACHEPREFIX="${RUNTIME_ROOT}/pycache"

mkdir -p \
    "${RUNTIME_ROOT}/tmp" \
    "${RUNTIME_ROOT}/pycache" \
    "${CHECKPOINT_BASE_DIR}" \
    "${LOG_DIR}"

echo "[INFO] Turbo root        = ${TURBO_ROOT}"
echo "[INFO] model repo        = ${MODEL_REPO}"
echo "[INFO] integration       = ${OPENPI_MODEL_DIR}"
echo "[INFO] initial weights   = ${WEIGHT_PATH}"
echo "[INFO] output root       = ${OUTPUT_ROOT}"
echo "[INFO] log directory     = ${LOG_DIR}"
echo "[INFO] runtime scratch   = ${RUNTIME_ROOT}"
cd "${MODEL_REPO}"
