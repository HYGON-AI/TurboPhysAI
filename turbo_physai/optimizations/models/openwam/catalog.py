# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenWAM main optimization ports; no source files are rewritten."""

from ....engine.definitions import group, wrap

_O = "turbo_physai.optimizations.models.openwam"

DATA = group(
    "openwam.data",
    wrap(
        "openwam.dataloader.robotwin._resolve_prompt",
        "turbo_physai.optimizations.models.openwam.ports.data._resolve_prompt_wrapper",
    ),
    wrap(
        "openwam.dataloader.robotwin.RoboTwinDataset._build_sample",
        "turbo_physai.optimizations.models.openwam.ports.data.RoboTwinDataset__build_sample_wrapper",
    ),
    wrap(
        "openwam.dataloader.robotwin.MultiTaskRoboTwinDataset.from_config",
        "turbo_physai.optimizations.models.openwam.ports.data.MultiTaskRoboTwinDataset_from_config_wrapper",
    ),
)

COMPUTE = group(
    "openwam.compute",
    wrap(
        "openwam.model.action_backbone.components.RMSNorm.forward",
        "turbo_physai.optimizations.models.openwam.ports.action_components.RMSNorm_forward_wrapper",
    ),
    wrap(
        "openwam.model.action_backbone.separate_action_dit.BridgeCrossAttention.forward",
        "turbo_physai.optimizations.models.openwam.ports.action.BridgeCrossAttention_forward_wrapper",
    ),
    wrap(
        "openwam.model.action_backbone.separate_action_dit.CrossAttnActionDiTBlock.forward",
        "turbo_physai.optimizations.models.openwam.ports.action.CrossAttnActionDiTBlock_forward_wrapper",
    ),
    wrap(
        "openwam.model.action_backbone.separate_action_dit.SelfAttnActionDiTBlock.forward",
        "turbo_physai.optimizations.models.openwam.ports.action.SelfAttnActionDiTBlock_forward_wrapper",
    ),
    wrap(
        "openwam.model.kernels.fused_blocks.act_post",
        "turbo_physai.optimizations.models.openwam.ports.action.act_post_wrapper",
    ),
    wrap(
        "openwam.model.video_backbone.wan.models.dit.flash_attention",
        "turbo_physai.optimizations.models.openwam.ports.dit.flash_attention_wrapper",
        aliases=(
            "openwam.model.video_backbone.wan.models.image_encoder.flash_attention",
        ),
    ),
    wrap(
        "openwam.model.video_backbone.wan.models.dit.RMSNorm.forward",
        "turbo_physai.optimizations.models.openwam.ports.dit.RMSNorm_forward_wrapper",
    ),
    wrap(
        "openwam.model.video_backbone.wan.models.dit.SelfAttention.forward",
        "turbo_physai.optimizations.models.openwam.ports.dit.SelfAttention_forward_wrapper",
    ),
    wrap(
        "openwam.model.video_backbone.wan.models.dit.CrossAttention.forward",
        "turbo_physai.optimizations.models.openwam.ports.dit.CrossAttention_forward_wrapper",
    ),
    wrap(
        "openwam.model.kernels.fused_blocks.masked_mha",
        "turbo_physai.optimizations.models.openwam.ports.wan.masked_mha_wrapper",
    ),
    wrap(
        "openwam.model.kernels.fused_blocks.wan_post",
        "turbo_physai.optimizations.models.openwam.ports.wan.wan_post_wrapper",
    ),
    wrap(
        "openwam.model.video_backbone.wan_backbone.WanBase.preprocess_input_for_train",
        "turbo_physai.optimizations.models.openwam.ports.wan.WanBase_preprocess_input_for_train_wrapper",
    ),
)

ARCHITECTURE = group(
    "openwam.architecture",
    wrap(
        "openwam.model.architectures.base.BaseWAMArchitecture._append_proprio_context_token",
        "turbo_physai.optimizations.models.openwam.ports.architecture.BaseWAMArchitecture__append_proprio_context_token_wrapper",
    ),
    wrap(
        "openwam.model.architectures.base.BaseWAMArchitecture.save_checkpoint",
        "turbo_physai.optimizations.models.openwam.ports.architecture.BaseWAMArchitecture_save_checkpoint_wrapper",
    ),
    wrap(
        "openwam.model.architectures.base.BaseWAMArchitecture.prepare_inputs",
        "turbo_physai.optimizations.models.openwam.ports.architecture.BaseWAMArchitecture_prepare_inputs_wrapper",
    ),
    wrap(
        "openwam.model.architectures.base.BaseWAMArchitecture.compute_loss",
        "turbo_physai.optimizations.models.openwam.ports.architecture.BaseWAMArchitecture_compute_loss_wrapper",
    ),
    wrap(
        "openwam.model.architectures.dual_system.idm.DualSystemIDMArchitecture.compute_loss",
        "turbo_physai.optimizations.models.openwam.ports.idm.DualSystemIDMArchitecture_compute_loss_wrapper",
    ),
    wrap(
        "openwam.model.architectures.dual_system.joint_self_attn.DualSystemSelfAttnArchitecture.forward",
        "turbo_physai.optimizations.models.openwam.ports.joint.DualSystemSelfAttnArchitecture_forward_wrapper",
    ),
)

MOT = group(
    "openwam.mot",
    wrap(
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._mixed_attention",
        "turbo_physai.optimizations.models.openwam.ports.mot.DualSystemMoTDriver__mixed_attention_wrapper",
    ),
    wrap(
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_checkpointed",
        "turbo_physai.optimizations.models.openwam.ports.mot.DualSystemMoTDriver__step_checkpointed_wrapper",
    ),
    wrap(
        "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.run_joint_loop",
        "turbo_physai.optimizations.models.openwam.ports.mot.DualSystemMoTDriver_run_joint_loop_wrapper",
    ),
    depends_on=("openwam.compute",),
)

TEXT = group(
    "openwam.text",
    wrap(
        "openwam.model.video_backbone.wan.encode.encode_text",
        "turbo_physai.optimizations.models.openwam.ports.encode.encode_text_wrapper",
    ),
)

TRAINING = group(
    "openwam.training",
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.__init__",
        "turbo_physai.optimizations.models.openwam.ports.trainer.trainer_init",
    ),
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.train",
        "turbo_physai.optimizations.models.openwam.ports.trainer.OpenWAMTrainer_train_wrapper",
    ),
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.build_optimizer",
        "turbo_physai.optimizations.models.openwam.ports.trainer.build_optimizer",
    ),
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.build_dataloader",
        "turbo_physai.optimizations.models.openwam.ports.trainer.build_dataloader",
    ),
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.prepare_accelerate",
        "turbo_physai.optimizations.models.openwam.ports.trainer.prepare_accelerate",
    ),
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.compute_loss",
        "turbo_physai.optimizations.models.openwam.ports.trainer.compute_loss",
    ),
    wrap(
        "openwam.train.openwam_trainer.OpenWAMTrainer.finish_training",
        "turbo_physai.optimizations.models.openwam.ports.trainer.finish_training",
    ),
    wrap(
        "openwam.train.utils.runtime_tuning._gpu_numa_cpus",
        "turbo_physai.optimizations.models.openwam.ports.trainer._gpu_numa_cpus_wrapper",
    ),
    wrap(
        "accelerate.utils.dataclasses.DeepSpeedPlugin.__post_init__",
        "turbo_physai.optimizations.models.openwam.launch.deepspeed_post_init",
        collect_target_hash=False,
    ),
    wrap(
        "accelerate.utils.dataclasses.DeepSpeedPlugin.set_mixed_precision",
        "turbo_physai.optimizations.models.openwam.launch.deepspeed_mixed_precision",
        collect_target_hash=False,
    ),
    depends_on=(
        "openwam.data",
        "openwam.compute",
        "openwam.architecture",
        "openwam.mot",
        "openwam.text",
    ),
)
