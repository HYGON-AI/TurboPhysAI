#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate mmcv2

cd "${POINTPILLARS_REPO_ROOT:-/workspace/PointPillars_TurboPhysAI/mmdetection3d}"
mkdir -p "${LOG_DIR:-/workspace/PointPillars_TurboPhysAI/logs}"

turbo-physai run \
    --optimization-config /workspace/PointPillars_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/configs/optimization.yaml \
    --runtime-config /workspace/PointPillars_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/configs/runtime.yaml \
    --set HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --set "PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}" \
    --log-report -- \
    torchrun --nproc_per_node 8 --standalone \
    tools/train.py "${POINTPILLARS_CONFIG:-configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py}" \
    --launcher pytorch \
    --work-dir "${WORK_DIR:-/workspace/PointPillars_TurboPhysAI/work_dirs/pointpillars_turbo_$(date +%Y%m%d_%H%M%S)}" \
    --cfg-options \
    train_dataloader.batch_size=4 \
    train_dataloader.num_workers=10 \
    default_hooks.logger.interval=1 \
    "train_dataloader.dataset.data_root=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/" \
    "val_dataloader.dataset.data_root=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/" \
    "test_dataloader.dataset.data_root=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/" \
    "val_evaluator.data_root=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/" \
    "val_evaluator.ann_file=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/nuscenes_infos_val.pkl" \
    "test_evaluator.data_root=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/" \
    "test_evaluator.ann_file=${DATA_ROOT:-/workspace/PointPillars_TurboPhysAI/static_data/nuscenes_pointpillars}/nuscenes_infos_val.pkl" \
    "$@" 2>&1 | tee "${LOG_DIR:-/workspace/PointPillars_TurboPhysAI/logs}/pointpillars_$(date +%Y%m%d_%H%M%S)_$$.log"
