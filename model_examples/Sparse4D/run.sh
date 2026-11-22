#!/usr/bin/env bash

# Derived from Sparse4D: tools/dist_train.sh
# Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
# Copyright (c) 2024 Horizon Robotics
# SPDX-License-Identifier: MIT
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon.

set -o pipefail

cd /workspace/Sparse4D || exit 1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TOPO_FILE=null
export NCCL_ALGO=Ring
export NCCL_SOCKET_IFNAME=ens47f0np0
export GLOO_SOCKET_IFNAME=ens47f0np0
export NCCL_RINGS="N0 0 7 6 5 4 3 2 1 N0|N1 1 2 3 4 5 6 7 0 N1|N2 2 1 0 7 6 5 4 3 N2|N3 3 4 5 6 7 0 1 2 N3|N4 4 3 2 1 0 7 6 5 N4|N5 5 6 7 0 1 2 3 4 N5|N6 6 5 4 3 2 1 0 7 N6|N7 7 0 1 2 3 4 5 6 N7"

turbo-physai run \
  --model Sparse4D \
  --log-report \
  -- python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=28652 \
    tools/train.py \
    /workspace/TurboPhysAI/model_examples/Sparse4D/sparse4dv4_bs18.py \
    --launcher pytorch \
    --work-dir=/workspace/work_dirs/sparse4dv4_bs18 \
    --cfg-options \
      data_root=/workspace/data/nuscenes/ \
      data.train.ann_file=/workspace/data/nuscenes_anno_pkls/nuscenes_infos_train.pkl \
      data.val.ann_file=/workspace/data/nuscenes_anno_pkls/nuscenes_infos_val.pkl \
      data.test.ann_file=/workspace/data/nuscenes_anno_pkls/nuscenes_infos_val.pkl \
      model.img_backbone.pretrained=/workspace/ckpt/resnet50-19c8e357.pth \
      log_config.interval=1 \
      2>&1 | tee /workspace/train.log
