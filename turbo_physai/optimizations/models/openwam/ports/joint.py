# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from openwam.model.architectures.dual_system.joint_self_attn import (
    logger,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.architectures.dual_system.joint_self_attn import (
        Optional,
        Tensor,
        Tuple,
    )


def DualSystemSelfAttnArchitecture_forward(
    self,
    noisy_actions: Optional[Tensor],
    action_timestep: Optional[Tensor],
    *,
    proprio: Optional[Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **pipeline_inputs,
) -> Tuple[Tensor, Optional[Tensor]]:
    vb = self.video_backbone
    ab = self.action_backbone
    if vb is None:
        raise RuntimeError(
            "video_backbone is None — pass pipe= to build_architecture or "
            "architecture.__init__ to enable forward()."
        )

    pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
    action_context = pipeline_inputs.get("context")
    action_context_mask = pipeline_inputs.get("context_mask")
    if (
        action_context is not None
        and action_context_mask is None
        and pipeline_inputs.get("seq_lens") is not None
    ):
        seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
        positions = torch.arange(action_context.shape[1], device=action_context.device)
        action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
    # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
    # path; TI2V itself fires that path first so these kwargs are inert for
    # it). VACE and I2V do NOT emit ``first_frame_latents`` (VACE routes
    # its first-frame condition through ``vace_context``; I2V uses the
    # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
    # structurally inert — kept on only so the joint MoT driver gets the
    # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
    # pass False.
    pipeline_inputs.setdefault("force_per_token_t_mod", True)
    pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
    vstate = vb.prepare(
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        **pipeline_inputs,
    )

    if noisy_actions is None or ab is None:
        from turbo_physai.optimizations.models.openwam.sac import active

        if self.training and active():
            raise ValueError(
                "Complete SAC requires the joint video/action training path"
            )
        from turbo_physai.optimizations.models.openwam import deferred_gather

        # Only the joint loop waits per layer for deferred_gather.
        deferred_gather.finish()
        for block_id in range(vb.num_layers):
            vstate = vb.run_block(block_id, vstate)
        return vb.finalize(vstate), None

    driver = self._mot_driver
    if driver is None:
        driver = self.build_mot_driver()

    astate = ab.prepare_state(
        noisy_actions,
        action_timestep,
        context=action_context,
        context_mask=action_context_mask,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
    if (
        compiled_loop is not None
        and not use_gradient_checkpointing
        and not use_gradient_checkpointing_offload
    ):
        from turbo_physai.optimizations.models.openwam import deferred_gather

        deferred_gather.finish()  # The compiled loop does not wait per layer.
        try:
            vstate, astate = compiled_loop(vstate, astate)
        except Exception as exc:
            self._compiled_mot_run_joint_loop = None
            logger.warning(
                "dual self-attn compiled MoT loop failed; falling back to eager: %s",
                exc,
            )
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
    else:
        vstate, astate = driver.run_joint_loop(
            vstate,
            astate,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
    return vb.finalize(vstate), ab.extract_prediction(astate)


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemSelfAttnArchitecture_forward_wrapper = _port_wrapper(
    DualSystemSelfAttnArchitecture_forward,
    "openwam.model.architectures.dual_system.joint_self_attn.DualSystemSelfAttnArchitecture.forward",
    "openwam.architecture",
)
