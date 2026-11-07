# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import functools
import logging
from openwam.model.kernels import joint_cat
from openwam.model.kernels.attention import sdpa
from openwam.model.architectures.utils.mask_modes import (
    build_cross_modal_attention_mask,
)
from ..attention import split_attention, split_mask_plan
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
    if enabled("mot_split_attention"):
        from openwam.model.video_backbone.wan.models.dit import flash_attention

        out = split_attention(
            q_cat, k_cat, v_cat, attn_mask, self.num_heads, flash_attention
        )
        if out is not None:
            return out
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
    compiled = None
    if enabled("global_compile"):
        from turbo_physai.optimizations.models.openwam.global_compile import (
            compile_scope,
            compile_mot_layer,
            mark_mot_text_dynamic,
        )

        if "mot" in compile_scope():
            if vstate.extras.get("vace") is not None:
                raise ValueError(
                    "MoT block compile currently requires the native Wan path without VACE"
                )
            mark_mot_text_dynamic(vstate, astate)
            if getattr(self, "_compiled_layer", None) is None:
                self._compiled_layer = compile_mot_layer(self)
            compiled = self._compiled_layer
            video_block = vstate.extras["dit"].blocks[layer_id]
            action_block = self.ab.blocks[layer_id]

    def _run(vx: Tensor, ax: Tensor) -> Tuple[Tensor, Tensor]:
        local_vstate = copy.copy(vstate)
        local_astate = copy.copy(astate)
        local_payload = copy.copy(outer_payload)
        local_astate.payload = local_payload
        local_vstate.hidden_states = vx
        local_payload.x_action = ax
        if compiled is None:
            self._step_impl(
                layer_id,
                local_vstate,
                local_astate,
                attn_mask=attn_mask,
                suppress_inner_attn_ckpt=True,
            )
        else:
            compiled(video_block, action_block, local_vstate, local_astate, attn_mask)
        return local_vstate.hidden_states, local_payload.x_action

    vx0 = vstate.hidden_states
    ax0 = outer_payload.x_action

    checkpoint_kwargs = {"use_reentrant": False}
    from turbo_physai.optimizations.models.openwam.sac import active, context_fn

    if active():
        if offload:
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


def DualSystemMoTDriver__build_attention_mask(
    self,
    s_video: int,
    s_action: int,
    video_tokens_per_frame: int,
    *,
    device: torch.device,
) -> Tensor:
    """Build the ``[Sv+Sa, Sv+Sa]`` cross-modal bool mask for the configured mode.

    Delegates to :func:`build_cross_modal_attention_mask`. ``v↔v`` follows
    ``vb.video_attention_mask_mode``; ``a↔a`` is fully connected; ``v↔a``
    follows ``attention_mask_mode`` (see :mod:`utils.mask_modes`).
    """
    split_enabled = enabled("mot_split_attention")
    mask = build_cross_modal_attention_mask(
        self.vb,
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=video_tokens_per_frame,
        mode=self.attention_mask_mode,
        device=torch.device("cpu") if split_enabled else device,
    )
    plan = split_mask_plan(mask) if split_enabled else None
    if enabled("host_async"):
        from turbo_physai.optimizations.models.openwam.host_async import upload

        mask = upload(mask, device)
    else:
        mask = mask.to(device=device)
    if plan is not None:
        try:
            version = mask._version
        except RuntimeError:  # Inference tensors have no version counter.
            return mask
        mask._openwam_split_plan = plan
        mask._openwam_split_version = version
    return mask


DualSystemMoTDriver__build_attention_mask_wrapper = _port_wrapper(
    DualSystemMoTDriver__build_attention_mask,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._build_attention_mask",
    "openwam.mot",
)


def DualSystemMoTDriver__step_impl(
    self,
    layer_id: int,
    vstate: "BlockLoopState",
    astate: "ActionState",
    attn_mask: Optional[Tensor] = None,
    *,
    suppress_inner_attn_ckpt: bool = False,
) -> Tuple["BlockLoopState", "ActionState"]:
    """Unwrapped per-layer body. See :meth:`step` for the public entry point.

    INVARIANT (consumed by :meth:`_step_checkpointed`): this method must
    only reassign the two layer-varying tensor fields ``vstate.hidden_states`` and
    ``astate.payload.x_action``. It MUST NOT mutate any layer-invariant
    field of ``vstate`` / ``astate`` in place — concretely, do not append
    to ``vace_hints``, write into ``extras``, or mutate ``context`` /
    ``freqs`` / ``t_mod``. ``_step_checkpointed`` runs this body inside
    ``torch.utils.checkpoint`` against shallow copies of the state
    objects; only ``x`` / ``x_action`` are isolated, everything else is
    shared by reference. In-place writes there would be re-applied on
    the backward recompute and silently corrupt the outer state without
    any test detecting it.
    """
    vb = self.vb
    ab = self.ab

    q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer_id, vstate)
    q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer_id, astate)

    if q_v.dtype != q_a.dtype:
        raise RuntimeError(
            f"DualSystemMoTDriver: dtype mismatch at layer {layer_id} "
            f"(video={q_v.dtype}, action={q_a.dtype}). Both backbones "
            "must produce attention inputs in matching dtype."
        )
    if q_v.device != q_a.device:
        raise RuntimeError(
            f"DualSystemMoTDriver: device mismatch at layer {layer_id} (video={q_v.device}, action={q_a.device})."
        )

    s_video = q_v.shape[1]
    if enabled("mot_split_attention") and not (
        self.mot_checkpoint_mixed_attn and ab.training and not suppress_inner_attn_ckpt
    ):
        from turbo_physai.optimizations.models.openwam.attention import (
            separate_attention_plan,
            split_attention_separate,
        )

        plan = separate_attention_plan(attn_mask, s_video, q_a.shape[1])
        if plan is not None:
            from openwam.model.video_backbone.wan.models.dit import flash_attention

            k_cat = joint_cat.cat_seq(k_v, k_a)
            v_cat = joint_cat.cat_seq(v_v, v_a)
            attn_v, attn_a = run_attention(
                split_attention_separate,
                q_v,
                None,
                None,
                q_a,
                None,
                None,
                plan,
                self.num_heads,
                flash_attention,
                joint_kv=(k_cat, v_cat),
                label="mixed_attention",
            )
            vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v, vpost)
            astate = ab.post_attn_at_layer(layer_id, astate, attn_a, apost)
            return vstate, astate

    # torch.cat([x_v, x_a], dim=1); one Triton copy pass with OPENWAM_FAST_CAT=1
    q_cat = joint_cat.cat_seq(q_v, q_a)
    k_cat = joint_cat.cat_seq(k_v, k_a)
    v_cat = joint_cat.cat_seq(v_v, v_a)
    # Contract: `run_joint_loop` pre-builds ``attn_mask`` once per forward and
    # passes it in for every layer. _step_impl does NOT rebuild the mask
    # itself; any direct caller must pass the same pre-built mask.

    if self.mot_checkpoint_mixed_attn and ab.training and not suppress_inner_attn_ckpt:
        mixed = torch.utils.checkpoint.checkpoint(
            self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
        )
    else:
        mixed = self._mixed_attention(q_cat, k_cat, v_cat, attn_mask)

    attn_v, attn_a = joint_cat.split_seq(
        mixed, s_video
    )  # split along dim 1, both parts contiguous
    vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v, vpost)
    astate = ab.post_attn_at_layer(layer_id, astate, attn_a, apost)
    return vstate, astate


DualSystemMoTDriver__step_impl_wrapper = _port_wrapper(
    DualSystemMoTDriver__step_impl,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_impl",
    "openwam.mot",
)


def DualSystemMoTDriver_step(
    self,
    layer_id: int,
    vstate: "BlockLoopState",
    astate: "ActionState",
    attn_mask: Optional[Tensor] = None,
    *,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
) -> Tuple["BlockLoopState", "ActionState"]:
    """Run one mixed-attention layer.

    Pulls Q/K/V from each backbone, concatenates along sequence,
    runs mixed attention (optionally checkpointed), splits the
    result, and feeds each slice through its post-attention suffix.

    ``attn_mask`` is the joint ``[Sv+Sa, Sv+Sa]`` mask built once at
    :meth:`run_joint_loop`. When ``None``, the driver falls back to
    per-step construction (used by direct ``step()`` callers in tests).

    When ``use_gradient_checkpointing`` is enabled (and the action
    backbone is in training mode), the entire pre→mixed→post triplet
    runs under :func:`torch.utils.checkpoint.checkpoint`, mirroring
    what the other architectures get from ``vb.run_block``. The inner
    per-layer ``mot_checkpoint_mixed_attn`` is suppressed in that mode
    to avoid nested-checkpoint waste — the outer wrapper already
    recomputes mixed attention.
    """
    if use_gradient_checkpointing and self.ab.training:
        return self._step_checkpointed(
            layer_id,
            vstate,
            astate,
            attn_mask=attn_mask,
            offload=use_gradient_checkpointing_offload,
        )
    if enabled("global_compile"):
        from turbo_physai.optimizations.models.openwam.global_compile import (
            compile_scope,
            compile_mot_layer,
            mark_mot_text_dynamic,
        )

        if "mot" in compile_scope():
            if vstate.extras.get("vace") is not None:
                raise ValueError(
                    "MoT block compile currently requires the native Wan path without VACE"
                )
            mark_mot_text_dynamic(vstate, astate)
            if getattr(self, "_compiled_layer", None) is None:
                self._compiled_layer = compile_mot_layer(self)
            return self._compiled_layer(
                vstate.extras["dit"].blocks[layer_id],
                self.ab.blocks[layer_id],
                vstate,
                astate,
                attn_mask,
            )
    return self._step_impl(layer_id, vstate, astate, attn_mask=attn_mask)


DualSystemMoTDriver_step_wrapper = _port_wrapper(
    DualSystemMoTDriver_step,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.step",
    "openwam.mot",
)
