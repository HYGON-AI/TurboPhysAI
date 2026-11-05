#!/usr/bin/env bash
set -euo pipefail

DIT4DIT_ROOT="${DIT4DIT_ROOT:-/workspace/model/DiT4DiT}"
COSMOS_MODEL_PATH="${COSMOS_MODEL_PATH:-/workspace/models/Cosmos-Predict2.5-2B-diffusers}"
DATA_ROOT="${DATA_ROOT:-/workspace/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim}"
RUN_ROOT="${RUN_ROOT:-/workspace/outputs}"
RUN_ID="${RUN_ID:-dit4dit_hcu_8card}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200000}"

cd "$DIT4DIT_ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"

turbo-physai run \
  --model dit4dit \
  --log-report \
  accelerate launch \
    --config_file DiT4DiT/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 8 \
    -m turbo_physai.optimizations.models.dit4dit.entry \
    --config_yaml DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml \
    --framework.name DiT4DiT \
    --framework.cosmos25.base_model "$COSMOS_MODEL_PATH" \
    --framework.action_model.action_model_type DiT-B \
    --framework.cosmos25.extract_layer 17 \
    --framework.cosmos25.flow_matching.time_distribution uniform \
    --framework.cosmos25.flow_matching.high_sigma_ratio null \
    --framework.cosmos25.flow_matching.high_sigma_min null \
    --framework.cosmos25.conditional_frame_timestep 0.0001 \
    --datasets.vla_data.data_root_dir "$DATA_ROOT" \
    --datasets.vla_data.data_mix fourier_gr1_unified_1000 \
    --datasets.vla_data.per_device_batch_size 4 \
    --datasets.vla_data.action_video_freq_ratio 2 \
    --trainer.freeze_modules \
      backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae \
    --trainer.max_train_steps "$MAX_TRAIN_STEPS" \
    --trainer.save_interval 50000 \
    --trainer.logging_frequency 10 \
    --trainer.eval_interval 100 \
    --trainer.learning_rate.backbone_interface 1e-5 \
    --trainer.learning_rate.action_model 1e-4 \
    --trainer.num_warmup_steps 5000 \
    --trainer.gradient_accumulation_steps 2 \
    --run_root_dir "$RUN_ROOT" \
    --run_id "$RUN_ID"
