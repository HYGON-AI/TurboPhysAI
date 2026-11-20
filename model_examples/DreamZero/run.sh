#!/usr/bin/env bash

set -o pipefail

cd /workspace/dreamzero || exit 1

export DREAMZERO_ROOT="${DREAMZERO_ROOT:-/workspace/dreamzero}"
export DROID_DATA_ROOT="${DROID_DATA_ROOT:-/workspace/datasets/droid_lerobot}"
export WAN22_CKPT_DIR="${WAN22_CKPT_DIR:-/workspace/checkpoints/Wan2.2-TI2V-5B}"
export IMAGE_ENCODER_DIR="${IMAGE_ENCODER_DIR:-/workspace/checkpoints/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/workspace/checkpoints/umt5-xxl}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/runs/dreamzero_droid_wan22_full}"
export HYDRA_FULL_ERROR=1

RUN_DIR="$OUTPUT_DIR/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/train.log"

turbo-physai run \
    --model DreamZero \
    --log-report \
    -- python3 -m torch.distributed.run --nproc_per_node 8 --standalone --local-ranks-filter 0 groot/vla/experiment/experiment.py \
    report_to=none \
    wandb_project=dreamzero \
    logging_steps=1 \
    data=dreamzero/droid_relative_wan22 \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    optim=adamw_apex_fused \
    training_args.deepspeed=/workspace/TurboPhysAI/model_examples/DreamZero/zero2_overlap.json \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$RUN_DIR \
    per_device_train_batch_size=1 \
    global_batch_size=8 \
    max_steps=200000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=false \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=160 \
    frame_seqlen=50 \
    save_lora_only=false \
    max_chunk_size=4 \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
  2>&1 | tee -a "$LOG_FILE"
