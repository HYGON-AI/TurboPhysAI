# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import copy
import logging
import types
import torch
import torch.nn as nn
import openwam.model.architectures.dual_system.mot_driver as _source
from turbo_physai.optimizations.common.python.source_binding import bind_missing_globals
from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.attention import (
    split_attention,
    split_mask_plan,
)
from turbo_physai.optimizations.models.openwam.sac import run_attention


def DualSystemMoTDriver___init__(
    self,
    vb: "VideoBackbone",
    ab: "ActionDiTBackbone",
    *,
    mot_checkpoint_mixed_attn: bool = True,
    attention_mask_mode: str = ACTION_SEES_VIDEO,
    video_attention_mask_mode: Optional[str] = None,
) -> None:
    if vb.num_layers != ab.num_layers:
        raise ValueError(
            f"DualSystemMoTDriver: video num_layers ({vb.num_layers}) must equal "
            f"action num_layers ({ab.num_layers}) for joint self-attention."
        )
    if vb.num_heads != ab.num_heads:
        raise ValueError(
            f"DualSystemMoTDriver: video num_heads ({vb.num_heads}) must equal "
            f"action num_heads ({ab.num_heads}). Per-head attention layout must match so "
            f"the concatenated Q/K/V can run through a single attention."
        )
    if vb.head_dim != ab.head_dim:
        raise ValueError(
            f"DualSystemMoTDriver: video head_dim ({vb.head_dim}) must equal action head_dim ({ab.head_dim})."
        )
    validate_attention_mask_mode(attention_mask_mode)

    self.vb = vb
    self.ab = ab
    self.num_layers = vb.num_layers
    self.num_heads = vb.num_heads
    self.head_dim = vb.head_dim
    self.mot_checkpoint_mixed_attn = bool(mot_checkpoint_mixed_attn)
    self.attention_mask_mode = attention_mask_mode
    self._compiled_layer = None

    # Allow the architecture / config to override the video v↔v sub-mode.
    # When None we defer to whatever ``vb.video_attention_mask_mode`` reports.
    set_video_attention_mask_mode(vb, video_attention_mask_mode)


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
    split_enabled = enabled("mot_split_attn")
    mask = build_cross_modal_attention_mask(
        self.vb,
        s_video=s_video,
        s_action=s_action,
        video_tokens_per_frame=video_tokens_per_frame,
        mode=self.attention_mask_mode,
        device=torch.device("cpu") if split_enabled else device,
    )
    plan = split_mask_plan(mask) if split_enabled else None
    mask = mask.to(device=device)
    if plan is not None:
        try:
            version = mask._version
        except RuntimeError:  # Inference tensors have no version counter.
            return mask
        mask._openwam_split_plan = plan
        mask._openwam_split_version = version
    return mask


def DualSystemMoTDriver__mixed_attention(
    self,
    q_cat: Tensor,
    k_cat: Tensor,
    v_cat: Tensor,
    attn_mask: Optional[Tensor],
) -> Tensor:
    """Single mixed self-attention over the concatenated [v, a] sequence.

    Inputs are ``(B, S, H*D)`` (the layout produced by ``pre_attn_at_layer``
    in both backbones). Output is the same layout. SDPA is used so an
    optional bool ``attn_mask`` (True = keep) can be honored.
    """
    return run_attention(
        functools.partial(DualSystemMoTDriver__mixed_attention_impl, self),
        q_cat,
        k_cat,
        v_cat,
        attn_mask,
        label="mixed_attention",
    )


def DualSystemMoTDriver__mixed_attention_impl(self, q_cat, k_cat, v_cat, attn_mask):
    if enabled("mot_split_attn"):
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
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return rearrange(out, "b n s d -> b s (n d)", n=n)


def DualSystemMoTDriver_step(
    self,
    layer_id: int,
    vstate: "BlockLoopState",
    astate: "ActionState",
    attn_mask: Optional[Tensor] = None,
    *,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    suppress_inner_attn_ckpt: bool = False,
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
    if suppress_inner_attn_ckpt and enabled("global_compile"):
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
            if self._compiled_layer is None:
                self._compiled_layer = compile_mot_layer(self)
            return self._compiled_layer(
                vstate.extras["dit"].blocks[layer_id],
                self.ab.blocks[layer_id],
                vstate,
                astate,
                attn_mask,
            )
    return self._step_impl(
        layer_id,
        vstate,
        astate,
        attn_mask=attn_mask,
        suppress_inner_attn_ckpt=suppress_inner_attn_ckpt,
    )


def DualSystemMoTDriver__step_impl(
    self,
    layer_id: int,
    vstate: "BlockLoopState",
    astate: "ActionState",
    attn_mask: Optional[Tensor] = None,
    *,
    suppress_inner_attn_ckpt: bool = False,
    _blocks=None,
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

    if _blocks is None:
        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer(layer_id, vstate)
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer(layer_id, astate)
    else:
        q_v, k_v, v_v, vpost = vb.pre_attn_at_layer_for_compile(
            layer_id, vstate, block=_blocks[0]
        )
        q_a, k_a, v_a, apost = ab.pre_attn_at_layer_for_compile(
            layer_id, astate, block=_blocks[1]
        )

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
    s_action = q_a.shape[1]
    q_cat = torch.cat([q_v, q_a], dim=1)
    k_cat = torch.cat([k_v, k_a], dim=1)
    v_cat = torch.cat([v_v, v_a], dim=1)
    # Contract: `run_joint_loop` pre-builds ``attn_mask`` once per forward and
    # passes it in for every layer. _step_impl does NOT rebuild the mask
    # itself; any direct caller must pass the same pre-built mask.

    if self.mot_checkpoint_mixed_attn and ab.training and not suppress_inner_attn_ckpt:
        mixed = torch.utils.checkpoint.checkpoint(
            self._mixed_attention, q_cat, k_cat, v_cat, attn_mask, use_reentrant=False
        )
    else:
        attention = (
            self._mixed_attention if _blocks is None else self._compiled_attention
        )
        mixed = attention(q_cat, k_cat, v_cat, attn_mask)

    attn_v, attn_a = mixed.split([s_video, s_action], dim=1)
    if _blocks is None:
        vstate = vb.post_attn_at_layer(layer_id, vstate, attn_v.contiguous(), vpost)
        astate = ab.post_attn_at_layer(layer_id, astate, attn_a.contiguous(), apost)
    else:
        vstate = vb.post_attn_at_layer_for_compile(
            layer_id, vstate, attn_v.contiguous(), vpost, block=_blocks[0]
        )
        astate = ab.post_attn_at_layer_for_compile(
            layer_id, astate, attn_a.contiguous(), apost, block=_blocks[1]
        )
    return vstate, astate


def DualSystemMoTDriver__step_checkpointed(
    self,
    layer_id: int,
    vstate: "BlockLoopState",
    astate: "ActionState",
    *,
    attn_mask: Optional[Tensor],
    offload: bool,
) -> Tuple["BlockLoopState", "ActionState"]:
    """Run :meth:`_step_impl` under ``torch.utils.checkpoint``.

    ``_step_impl`` mutates ``vstate.hidden_states`` and ``astate.payload.x_action``
    as it walks the block, which is fine on forward but lethal on
    backward: ``torch.utils.checkpoint`` re-runs the closure during
    recompute, and that second mutation would clobber the post-forward
    value held by the outer state objects, leaving them pointing at a
    recomputed activation from an earlier layer once backward unwinds.

    We avoid that by giving the closure shallow copies of the state
    containers (``copy.copy`` on the dataclass / payload — same field
    references, fresh wrappers). Layer-invariant fields like
    ``context`` / ``freqs`` / ``t_mod`` / ``vace_hints`` / ``extras``
    are still shared by reference (cheap), but the two tensor fields
    the body writes (``x`` / ``x_action``) live on the local copies so
    recompute never touches the outer references the caller and the
    autograd graph hold. After the checkpoint returns we propagate the
    new tensor values onto the outer ``vstate`` / ``astate`` for the
    next layer's iteration in ``run_joint_loop``.
    """
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
            if self._compiled_layer is None:
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

    if active() or enabled("checkpoint_stats"):
        if offload and active():
            raise ValueError(
                "Complete SAC requires gradient checkpoint offload to be disabled"
            )
        checkpoint_kwargs["context_fn"] = context_fn(layer_id, self.num_layers)

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
) -> Tuple["BlockLoopState", "ActionState"]:
    """Run the full per-layer loop over both modalities.

    Builds the joint attention mask once from the initial shapes and
    reuses it across all layers (token counts and tokens-per-frame are
    layer-invariant during one forward). The two ``use_gradient_*``
    flags are forwarded to :meth:`step` so each layer may opt into
    step-level activation checkpointing — see the class docstring of
    :meth:`step` for memory/compute trade-offs.
    """
    # Resolve sequence shapes from the backbone-populated f/h/w fields.
    # ``vstate.hidden_states.shape[1]`` is identical to ``f*tokens_per_frame`` for
    # backbones that carry a 3D ``(B, S, D)`` state (Wan), but for
    # backbones whose ``state.hidden_states`` is natively 5D ``(B, T, H, W, D)``
    # (CosmosPredict25) ``shape[1]`` is just ``T`` — wrong. Going through f and
    # the shared ``compute_video_tokens_per_frame`` helper is the only
    # formulation that works for both layouts.
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

    # Backbones may prepend prefix K/V tokens (keys without matching query
    # rows — e.g. Cosmos3's cached und text stream). ``_step_impl`` needs no
    # change: it splits the attention output by query lengths, and SDPA
    # handles the resulting rectangular mask.
    attn_mask = widen_mask_for_prefix_kv(attn_mask, vstate)

    from turbo_physai.optimizations.models.openwam.runtime import checkpoint_layers

    checkpoint_count = checkpoint_layers(self.num_layers)
    partial = enabled("checkpoint_skip_layers")
    from turbo_physai.optimizations.models.openwam.sac import (
        active,
        layer_range,
        validate,
    )

    if active():
        validate(self.num_layers)
        if self.ab.training and not use_gradient_checkpointing:
            raise ValueError(
                "Complete SAC requires training.use_gradient_checkpointing=True"
            )
        sac_range = layer_range(self.num_layers)
        if getattr(self, "_reported_sac_layers", None) != sac_range:
            logging.getLogger(__name__).info(
                "[optimizations] standard SAC covers MoT layers [%d, %d) of %d; "
                "the no-checkpoint tail of %d layers keeps autograd activations",
                *sac_range,
                self.num_layers,
                self.num_layers - checkpoint_count,
            )
            self._reported_sac_layers = sac_range
    if (
        partial
        and getattr(self, "_reported_checkpoint_count", None) != checkpoint_count
    ):
        logging.getLogger(__name__).info(
            "[optimizations] checkpointed layers=%d/%d, no-checkpoint tail=%d, outer checkpoint=%s; "
            "tail layers have no inner attention checkpoint",
            checkpoint_count,
            self.num_layers,
            self.num_layers - checkpoint_count,
            bool(use_gradient_checkpointing and self.ab.training),
        )
        self._reported_checkpoint_count = checkpoint_count
    from turbo_physai.optimizations.models.openwam import deferred_gather

    for layer_id in range(self.num_layers):
        if deferred_gather.PENDING is not None:
            # deferred_gather: this layer's parameters may still be in flight.
            deferred_gather.PENDING.wait_layer(layer_id)
        vstate, astate = self.step(
            layer_id,
            vstate,
            astate,
            attn_mask=attn_mask,
            use_gradient_checkpointing=use_gradient_checkpointing
            and layer_id < checkpoint_count,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **(
                {"suppress_inner_attn_ckpt": True}
                if partial and layer_id >= checkpoint_count
                else {}
            ),
        )
    if deferred_gather.PENDING is not None:
        deferred_gather.PENDING.finish()
    return vstate, astate


bind_missing_globals(globals(), _source)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver___init___wrapper = _port_wrapper(
    DualSystemMoTDriver___init__,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.__init__",
    "openwam.mot",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver__build_attention_mask_wrapper = _port_wrapper(
    DualSystemMoTDriver__build_attention_mask,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._build_attention_mask",
    "openwam.mot",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver__mixed_attention_wrapper = _port_wrapper(
    DualSystemMoTDriver__mixed_attention,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._mixed_attention",
    "openwam.mot",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver_step_wrapper = _port_wrapper(
    DualSystemMoTDriver_step,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.step",
    "openwam.mot",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver__step_impl_wrapper = _port_wrapper(
    DualSystemMoTDriver__step_impl,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_impl",
    "openwam.mot",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver__step_checkpointed_wrapper = _port_wrapper(
    DualSystemMoTDriver__step_checkpointed,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver._step_checkpointed",
    "openwam.mot",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

DualSystemMoTDriver_run_joint_loop_wrapper = _port_wrapper(
    DualSystemMoTDriver_run_joint_loop,
    "openwam.model.architectures.dual_system.mot_driver.DualSystemMoTDriver.run_joint_loop",
    "openwam.mot",
)
