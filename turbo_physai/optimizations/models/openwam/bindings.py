# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Installed callables used to discover committed Turbo Group options."""

PATCH_POINTS = (
    ("openwam.dataloader.robotwin._resolve_prompt", "openwam.data"),
    ("openwam.dataloader.robotwin.RoboTwinDataset._build_sample", "openwam.data"),
    (
        "openwam.dataloader.robotwin.MultiTaskRoboTwinDataset.from_config",
        "openwam.data",
    ),
    ("openwam.model.action_backbone.components.rope_apply_1d", "openwam.compute"),
    ("openwam.model.action_backbone.components.RMSNorm.forward", "openwam.compute"),
    (
        "openwam.model.action_backbone.separate_action_dit.BridgeCrossAttention.forward",
        "openwam.compute",
    ),
    (
        "openwam.model.action_backbone.separate_action_dit.CrossAttnActionDiTBlock.forward",
        "openwam.compute",
    ),
    (
        "openwam.model.action_backbone.separate_action_dit.SelfAttnActionDiTBlock.gate",
        "openwam.compute",
    ),
    (
        "openwam.model.action_backbone.separate_action_dit.SelfAttnActionDiTBlock.forward",
        "openwam.compute",
    ),
    (
        "openwam.model.action_backbone.separate_action_dit.ActionDiT.pre_attn_at_layer_for_compile",
        "openwam.compute",
    ),
    (
        "openwam.model.action_backbone.separate_action_dit.ActionDiT.post_attn_at_layer_for_compile",
        "openwam.compute",
    ),
    (
        "openwam.model.architectures.base.BaseWAMArchitecture._append_proprio_context_token",
        "openwam.architecture",
    ),
    (
        "openwam.model.architectures.base.BaseWAMArchitecture.save_checkpoint",
        "openwam.architecture",
    ),
    (
        "openwam.model.architectures.base.BaseWAMArchitecture.prepare_inputs",
        "openwam.architecture",
    ),
    (
        "openwam.model.architectures.base.BaseWAMArchitecture.compute_loss",
        "openwam.architecture",
    ),
    (
        "openwam.model.architectures.dual_system.idm.DualSystemIDMArchitecture.compute_loss",
        "openwam.architecture",
    ),
    (
        "openwam.model.architectures.dual_system.joint_self_attn.DualSystemSelfAttnArchitecture.forward",
        "openwam.architecture",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.__init__",
        "openwam.mot",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._build_attention_mask",
        "openwam.mot",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._mixed_attention",
        "openwam.mot",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.step",
        "openwam.mot",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_impl",
        "openwam.mot",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_checkpointed",
        "openwam.mot",
    ),
    (
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.run_joint_loop",
        "openwam.mot",
    ),
    ("openwam.model.video_backbone.wan.encode.encode_text", "openwam.text"),
    ("openwam.model.video_backbone.wan.models.dit.flash_attention", "openwam.compute"),
    ("openwam.model.video_backbone.wan.models.dit.modulate", "openwam.compute"),
    ("openwam.model.video_backbone.wan.models.dit.rope_apply", "openwam.compute"),
    ("openwam.model.video_backbone.wan.models.dit.RMSNorm.forward", "openwam.compute"),
    (
        "openwam.model.video_backbone.wan.models.dit.SelfAttention.forward",
        "openwam.compute",
    ),
    (
        "openwam.model.video_backbone.wan.models.dit.CrossAttention.forward",
        "openwam.compute",
    ),
    (
        "openwam.model.video_backbone.wan.models.dit.GateModule.forward",
        "openwam.compute",
    ),
    (
        "openwam.model.video_backbone.wan.models.dit.DiTBlock.__init__",
        "openwam.compute",
    ),
    ("openwam.model.video_backbone.wan.models.vae.CausalConv3d.forward", "openwam.vae"),
    ("openwam.model.video_backbone.wan.models.vae.RMS_norm.forward", "openwam.vae"),
    ("openwam.model.video_backbone.wan.models.vae.Resample.forward", "openwam.vae"),
    (
        "openwam.model.video_backbone.wan.models.vae.ResidualBlock.forward",
        "openwam.vae",
    ),
    ("openwam.model.video_backbone.wan.models.vae.Encoder3d.forward", "openwam.vae"),
    ("openwam.model.video_backbone.wan.models.vae.Encoder3d_38.forward", "openwam.vae"),
    ("openwam.model.video_backbone.wan.models.vae.Decoder3d.forward", "openwam.vae"),
    ("openwam.model.video_backbone.wan.models.vae.Decoder3d_38.forward", "openwam.vae"),
    (
        "openwam.model.video_backbone.wan.models.vae.WanVideoVAE.batch_encode",
        "openwam.vae",
    ),
    (
        "openwam.model.video_backbone.wan.models.vae.WanVideoVAE.single_encode",
        "openwam.vae",
    ),
    (
        "openwam.model.video_backbone.wan.models.vae.WanVideoVAE.single_decode",
        "openwam.vae",
    ),
    ("openwam.model.video_backbone.wan.models.vae.VideoVAE38_.encode", "openwam.vae"),
    ("openwam.model.video_backbone.wan.preprocess.preprocess_video", "openwam.data"),
    ("openwam.model.video_backbone.wan_backbone.WanBase.prepare", "openwam.compute"),
    (
        "openwam.model.video_backbone.wan_backbone.WanBase.pre_attn_at_layer_for_compile",
        "openwam.compute",
    ),
    (
        "openwam.model.video_backbone.wan_backbone.WanBase.post_attn_at_layer_for_compile",
        "openwam.compute",
    ),
    (
        "openwam.model.video_backbone.wan_backbone.WanBase.preprocess_input_for_train",
        "openwam.compute",
    ),
    ("openwam.train.openwam_trainer.OpenWAMTrainer.__init__", "openwam.training"),
    ("openwam.train.openwam_trainer.OpenWAMTrainer.train", "openwam.training"),
    (
        "openwam.train.openwam_trainer.OpenWAMTrainer.build_optimizer",
        "openwam.training",
    ),
    (
        "openwam.train.openwam_trainer.OpenWAMTrainer.build_dataloader",
        "openwam.training",
    ),
    (
        "openwam.train.openwam_trainer.OpenWAMTrainer.prepare_accelerate",
        "openwam.training",
    ),
    ("openwam.train.openwam_trainer.OpenWAMTrainer.compute_loss", "openwam.training"),
    ("openwam.train.openwam_trainer.OpenWAMTrainer.log_step", "openwam.training"),
    (
        "openwam.train.openwam_trainer.OpenWAMTrainer.finish_training",
        "openwam.training",
    ),
)
