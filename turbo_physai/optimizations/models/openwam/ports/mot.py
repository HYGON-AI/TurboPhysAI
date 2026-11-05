# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import functools
import logging
import torch
from einops import rearrange
from openwam.model.architectures.base import validate_gradient_checkpointing_layers
from openwam.model.architectures.dual_system.mot_driver import (
    widen_mask_for_prefix_kv,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.architectures.dual_system.mot_driver import (
        ActionState,
        BlockLoopState,
        Optional,
        Tensor,
        Tuple,
    )
from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.sac import run_attention

logger = logging.getLogger(__name__)


def DualSystemMoTDriver__mixed_attention(
    self,
    q_cat: Tensor,
    k_cat: Tensor,
    v_cat: Tensor,
    attn_mask: Optional[Tensor],
) -> Tensor:
    """Single mixed self-attention over the concatenated [v, a] sequence."""
    return run_attention(
        functools.partial(DualSystemMoTDriver__mixed_attention_impl, self),
        q_cat,
        k_cat,
        v_cat,
        attn_mask,
        label="mixed_attention",
    )


def DualSystemMoTDriver__mixed_attention_impl(self, q_cat, k_cat, v_cat, attn_mask):
    from openwam.model.kernels.attention import sdpa

    n = self.num_heads
    q = rearrange(q_cat, "b s (n d) -> b n s d", n=n)
    k = rearrange(k_cat, "b s (n d) -> b n s d", n=n)
    v = rearrange(v_cat, "b s (n d) -> b n s d", n=n)
    out = sdpa(q, k, v, attn_mask=attn_mask)
    return rearrange(out, "b n s d -> b s (n d)", n=n)


def DualSystemMoTDriver__step_checkpointed(
    self,
    layer_id: int,
    vstate: "BlockLoopState",
    astate: "ActionState",
    *,
    attn_mask: Optional[Tensor],
    offload: bool,
) -> Tuple["BlockLoopState", "ActionState"]:
    """Run :meth:`_step_impl` under ``torch.utils.checkpoint`` with optional SAC context."""
    outer_payload = astate.payload

    def _run(vx: Tensor, ax: Tensor) -> Tuple[Tensor, Tensor]:
        local_vstate = copy.copy(vstate)
        local_astate = copy.copy(astate)
        local_payload = copy.copy(outer_payload)
        local_astate.payload = local_payload
        local_vstate.hidden_states = vx
        local_payload.x_action = ax
        self._step_impl(
            layer_id,
            local_vstate,
            local_astate,
            attn_mask=attn_mask,
            suppress_inner_attn_ckpt=True,
        )
        return local_vstate.hidden_states, local_payload.x_action

    vx0 = vstate.hidden_states
    ax0 = outer_payload.x_action

    checkpoint_kwargs = {"use_reentrant": False}
    from turbo_physai.optimizations.models.openwam.sac import active, context_fn

    if active():
        if offload and active():
            raise ValueError(
                "Complete SAC requires gradient checkpoint offload to be disabled"
            )
        n_ckpt = getattr(self, "_gradient_checkpointing_layers", None)
        checkpoint_kwargs["context_fn"] = context_fn(layer_id, self.num_layers, n_ckpt)

    if offload:
        with torch.autograd.graph.save_on_cpu():
            new_vx, new_ax = torch.utils.checkpoint.checkpoint(
                _run, vx0, ax0, **checkpoint_kwargs
            )
    else:
        new_vx, new_ax = torch.utils.checkpoint.checkpoint(
            _run, vx0, ax0, **checkpoint_kwargs
        )

    vstate.hidden_states = new_vx
    outer_payload.x_action = new_ax
    return vstate, astate


def DualSystemMoTDriver_run_joint_loop(
    self,
    vstate: "BlockLoopState",
    astate: "ActionState",
    *,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    gradient_checkpointing_layers: Optional[int] = None,
) -> Tuple["BlockLoopState", "ActionState"]:
    """Run the full per-layer loop over both modalities with deferred gather and SAC support."""
    requested_layers = validate_gradient_checkpointing_layers(
        gradient_checkpointing_layers
    )
    n_ckpt = (
        self.num_layers
        if requested_layers is None
        else min(self.num_layers, requested_layers)
    )
    self._gradient_checkpointing_layers = n_ckpt

    s_video = int(vstate.grid_frames) * self._video_tokens_per_frame(vstate)
    payload = astate.payload
    if payload is None or not hasattr(payload, "x_action"):
        raise RuntimeError(
            "DualSystemMoTDriver: astate.payload must expose `x_action` (populated by ActionDiT.prepare_state)."
        )
    s_action = payload.x_action.shape[1]

    attn_mask = self._build_attention_mask(
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=self._video_tokens_per_frame(vstate),
        device=vstate.hidden_states.device,
    )
    attn_mask = widen_mask_for_prefix_kv(attn_mask, vstate)

    from turbo_physai.optimizations.models.openwam.sac import (
        active,
        layer_range,
        validate,
    )

    if active():
        validate(self.num_layers, n_ckpt)
        if self.ab.training and not use_gradient_checkpointing:
            raise ValueError(
                "Complete SAC requires training.use_gradient_checkpointing=True"
            )
        sac_range = layer_range(self.num_layers, n_ckpt)
        if getattr(self, "_reported_sac_layers", None) != sac_range:
            logging.getLogger(__name__).info(
                "[optimizations] standard SAC covers MoT layers [%d, %d) of %d; "
                "the no-checkpoint tail of %d layers keeps autograd activations",
                *sac_range,
                self.num_layers,
                self.num_layers - n_ckpt,
            )
            self._reported_sac_layers = sac_range

    from turbo_physai.optimizations.models.openwam import deferred_gather

    for layer_id in range(self.num_layers):
        if deferred_gather.PENDING is not None:
            deferred_gather.PENDING.wait_layer(layer_id)
        vstate, astate = self.step(
            layer_id,
            vstate,
            astate,
            attn_mask=attn_mask,
            use_gradient_checkpointing=use_gradient_checkpointing and layer_id < n_ckpt,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
    if deferred_gather.PENDING is not None:
        deferred_gather.PENDING.finish()
    return vstate, astate


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver__mixed_attention_wrapper = _port_wrapper(
    DualSystemMoTDriver__mixed_attention,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._mixed_attention",
    "openwam.mot",
)

DualSystemMoTDriver__step_checkpointed_wrapper = _port_wrapper(
    DualSystemMoTDriver__step_checkpointed,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_checkpointed",
    "openwam.mot",
)

DualSystemMoTDriver_run_joint_loop_wrapper = _port_wrapper(
    DualSystemMoTDriver_run_joint_loop,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.run_joint_loop",
    "openwam.mot",
)
