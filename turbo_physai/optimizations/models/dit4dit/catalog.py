# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""DiT4DiT groups use the ordinary TurboPhysAI replace/wrap lifecycle."""

from turbo_physai import group, replace, wrap

P = "turbo_physai.optimizations.models.dit4dit"
C = "DiT4DiT.model.modules.vlm.Cosmos25"
T = "DiT4DiT.training.train"
CHECK = P + ".compatibility.baseline"

DATALOADER = group(
    "dit4dit.dataloader",
    wrap("DiT4DiT.dataloader.build_dataloader", P + ".dataloader.build_dataloader_wrapper",
         aliases=(T + ".build_dataloader",)),
    wrap(T + ".prepare_data", P + ".dataloader.prepare_data_wrapper"),
    compatibility_check=CHECK,
)
ATTENTION = group(
    "dit4dit.cosmos.attention",
    replace("diffusers.models.transformers.transformer_cosmos.CosmosAttnProcessor2_0.__call__",
            P + ".cosmos_attention.CosmosAttnProcessor2_0SkipNoopRepeat.__call__"),
    compatibility_check=CHECK,
)
CACHE = group(
    "dit4dit.cosmos.cache",
    wrap(C + ".Cosmos25FeatureExtractor", P + ".cosmos_cache.extractor_cache_wrapper"),
    compatibility_check=CHECK,
)
VAE_BATCH = group(
    "dit4dit.cosmos.vae_batch_encode",
    replace(C + ".Cosmos25FeatureExtractor.prepare_latents", P + ".cosmos_forward.prepare_latents"),
    depends_on=("dit4dit.cosmos.cache",),
    compatibility_check=CHECK,
)
HIDDEN_CAPTURE = group(
    "dit4dit.cosmos.hidden_capture",
    replace(C + ".Cosmos25FeatureExtractor.forward", P + ".cosmos_forward.feature_extractor_forward"),
    replace(C + ".Cosmos25FeatureExtractor._register_hook", P + ".cosmos_forward.register_direct_capture"),
    depends_on=("dit4dit.cosmos.cache",),
    compatibility_check=CHECK,
)
SEGMENT_COMPILE = group(
    "dit4dit.cosmos.segment_compile",
    # This is the eager adapter installed by hidden_capture. Patching its
    # factory keeps compilation independent of hidden capture and reversible.
    wrap(P + ".cosmos_forward.install_segmented_cosmos_forward", P + ".cosmos_forward.compile_segment_wrapper"),
    depends_on=("dit4dit.cosmos.hidden_capture",),
    compatibility_check=CHECK,
)
VIDEO_PACK = group(
    "dit4dit.cosmos.video_pack",
    replace(C + "._Cosmos25_Interface.build_cosmos_inputs", P + ".video_pack.build_cosmos_inputs"),
    wrap(P + ".video_pack.pack_flow_matching_video", P + ".video_pack.compile_pack_wrapper"),
    depends_on=("dit4dit.cosmos.hidden_capture",),
    compatibility_check=CHECK,
)
ACTION = group(
    "dit4dit.action.compile_rng_fallback",
    wrap("DiT4DiT.model.framework.build_framework", P + ".action_compile.build_framework_wrapper",
         aliases=(T + ".build_framework",)),
    compatibility_check=CHECK,
)
ADAMW = group(
    "dit4dit.training.fused_adamw",
    replace(T + ".setup_optimizer_and_scheduler", P + ".training.setup_optimizer_and_scheduler"),
    compatibility_check=CHECK,
)
METRICS = group(
    "dit4dit.training.metric_sync",
    replace(T + ".VLATrainer._train_step", P + ".training._train_step"),
    compatibility_check=CHECK,
)
