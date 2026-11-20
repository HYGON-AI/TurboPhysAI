"""dreamzero optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.
"""

from turbo_physai import group, replace, wrap  # noqa: F401


# Add validated model-specific Group declarations below.
# CausalWanModel (DiT): cancel gradient checkpointing + channels_last patch embedding
CAUSAL_WAN_MODEL_CHANNELS_LAST_GRADIENT_CHECKPOINTING = group(
    "dreamzero.dit.channels_last_gradient_checkpointing",
    replace(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanModel._forward_train"
        ),
        replacement="turbo_physai.optimizations.models.dreamzero.CausalWanModel.forward_train",
    ),
    replace(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanModel.init_weights"
        ),
        replacement="turbo_physai.optimizations.models.dreamzero.CausalWanModel.init_weights",
    ),
)

# CausalWanModel (DiT): fold independent state-attention blocks into the batch
# dim for one attention call per group of blocks.
CAUSAL_WAN_MODEL_FOLD_STATE_BLOCKS = group(
    "dreamzero.dit.fold_state_blocks",
    replace(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanSelfAttention._process_state_blocks"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.CausalWanSelfAttention.process_state_blocks"
        ),
    ),
)

# DiT cross-attention imports flash_attention by value, so replace its alias too.
DIT_DENSE_FLASH_ATTENTION = group(
    "dreamzero.dit.dense_flash_attention",
    wrap(
        target="groot.vla.model.dreamzero.modules.attention.flash_attention",
        aliases=("groot.vla.model.dreamzero.modules.wan2_1_submodule.flash_attention",),
        replacement="turbo_physai.optimizations.models.dreamzero.attention.dense_flash_attention",
    ),
)

# Real RoPE, compiled tensor regions, and shared noisy image/action Q/KV.
CAUSAL_WAN_MODEL_REAL_ROPE_COMPILE = group(
    "dreamzero.dit.real_rope_compile",
    wrap(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanModel.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.CausalWanModel.compile_real_rope_freqs"
        ),
    ),
    wrap(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanSelfAttention.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.CausalWanSelfAttention.compile_forward"
        ),
    ),
    wrap(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanAttentionBlock.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.CausalWanAttentionBlock.compile_forward"
        ),
    ),
    wrap(
        target=(
            "groot.vla.model.dreamzero.modules."
            "wan_video_dit_action_casual_chunk.CausalWanAttentionBlock.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.CausalWanAttentionBlock.compile_cross_attn"
        ),
    ),
    depends_on=("dreamzero.dit.dense_flash_attention",),
)

# Wan video VAE: channels_last for Resample/AttentionBlock/encode and WanVideoVAE38 conv weights
WAN_VIDEO_VAE_CHANNELS_LAST = group(
    "dreamzero.vae.channels_last",
    replace(
        target="groot.vla.model.dreamzero.modules.wan_video_vae.Resample.forward",
        replacement="turbo_physai.optimizations.models.dreamzero.wan_video_vae.resample_forward",
    ),
    replace(
        target="groot.vla.model.dreamzero.modules.wan_video_vae.AttentionBlock.forward",
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.wan_video_vae.attention_block_forward"
        ),
    ),
    replace(
        target="groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE.encode",
        replacement="turbo_physai.optimizations.models.dreamzero.wan_video_vae.encode",
    ),
    wrap(
        target="groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE38.__init__",
        replacement="turbo_physai.optimizations.models.dreamzero.wan_video_vae.init_channels_last",
    ),
)

# Wan video VAE: torch.compile each per-chunk Encoder3d/Encoder3d_38 forward
WAN_VIDEO_VAE_COMPILE_ENCODE = group(
    "dreamzero.vae.compile_encode",
    wrap(
        target="groot.vla.model.dreamzero.modules.wan_video_vae.Encoder3d.forward",
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.wan_video_vae.compile_encoder3d_forward"
        ),
    ),
    wrap(
        target="groot.vla.model.dreamzero.modules.wan_video_vae.Encoder3d_38.forward",
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.wan_video_vae.compile_encoder3d_38_forward"
        ),
    ),
)

# skip wan2.2 5B VAE: skips the VAE encoder and decoder for WAN2.2B, which is unused in DiT
SKIP_UNUSED_VAE = group(
    "dreamzero.training.skip_unused_vae",
    replace(
        target=(
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf."
            "WANPolicyHead.encode_image"
        ),
        replacement="turbo_physai.optimizations.models.dreamzero.skip_unused_vae.encode_image",
    ),
    replace(
        target=(
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf."
            "WANPolicyHead.forward"
        ),
        replacement="turbo_physai.optimizations.models.dreamzero.skip_unused_vae.forward",
    ),
)

# Wan text encoder (T5): SDPA attention kernel + torch.compile'd forward
WAN_VIDEO_TEXT_ENCODER = group(
    "dreamzero.text_encoder.wan_video_text_encoder",
    replace(
        target="groot.vla.model.dreamzero.modules.wan_video_text_encoder.T5Attention.forward",
        replacement="turbo_physai.optimizations.models.dreamzero.text_encoder.t5_attention_forward",
    ),
    wrap(
        target="groot.vla.model.dreamzero.modules.wan_video_text_encoder.WanTextEncoder.forward",
        replacement="turbo_physai.optimizations.models.dreamzero.text_encoder.compile_forward",
    ),
)

# Wan image encoder (CLIP ViT): torch.compile the VisionTransformer forward
WAN_IMAGE_ENCODER_COMPILE_FORWARD = group(
    "dreamzero.image_encoder.compile_forward",
    wrap(
        target="groot.vla.model.dreamzero.modules.wan_video_image_encoder.VisionTransformer.forward",
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.wan_video_image_encoder.compile_forward"
        ),
    ),
)

# train mode cache
TRAIN_MODE_CACHE = group(
    "dreamzero.training.cache_train_mode",
    wrap(
        target="groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead.train",
        replacement="turbo_physai.optimizations.models.dreamzero.train_mode_cache.train_wrapper",
    ),
    replace(
        target=(
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf."
            "WANPolicyHead.set_frozen_modules_to_eval_mode"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.train_mode_cache.set_frozen_modules_to_eval_mode"
        ),
    ),
    wrap(
        target=(
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf."
            "WANPolicyHead.set_trainable_parameters"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.train_mode_cache.invalidate_wrapper"
        ),
    ),
    wrap(
        target=(
            "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf."
            "WANPolicyHead.inject_lora_after_loading"
        ),
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.train_mode_cache.invalidate_wrapper"
        ),
    ),
)

FREEZE_GC_AFTER_WARMUP = group(
    "dreamzero.training.freeze_gc_after_warmup",
    wrap(
        target="groot.vla.experiment.experiment.VLATrainer.training_step",
        replacement=(
            "turbo_physai.optimizations.models.dreamzero.Fix_GC.freeze_gc_after_warmup"
        ),
    ),
)
