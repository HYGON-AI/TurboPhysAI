# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import torch
from torch import nn
from openwam.model.kernels.rope_triton import _HAVE_TRITON, _STATE, fused_rope_apply
from einops import rearrange
from openwam.model.video_backbone.wan.models.dit import (
    CrossAttention,
    GateModule,
    SelfAttention,
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
)
import openwam.model.video_backbone.wan.models.dit as _source
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.video_backbone.wan.models.dit import (
        Optional,
    )
from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.attention import fa2_padding
from turbo_physai.optimizations.models.openwam.linear_bias2d import video_linear
from turbo_physai.optimizations.models.openwam.sac import (
    dispatcher,
    run_attention,
    run_norm,
    run_projection,
)


@dispatcher
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    compatibility_mode=False,
    attn_mask: Optional[torch.Tensor] = None,
):
    return run_attention(
        _flash_attention_impl,
        q,
        k,
        v,
        num_heads,
        compatibility_mode=compatibility_mode,
        attn_mask=attn_mask,
    )


def _flash_attention_impl(q, k, v, num_heads, compatibility_mode=False, attn_mask=None):
    # FA2/FA3/sage are CUDA half-precision kernels; they raise on anything else.
    fused_ok = q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
    if not compatibility_mode:
        padded = fa2_padding(q, k, v, attn_mask, num_heads)
        if padded is not None:
            return padded
    if compatibility_mode or attn_mask is not None or not fused_ok:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = _source.flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x, tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = _source.flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = _source.sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def RMSNorm_forward(self, x):
    from turbo_physai.optimizations.models.openwam import enabled

    if enabled("pointwise_compile") or enabled("lightop_norm"):
        from turbo_physai.optimizations.models.openwam import pointwise

        return pointwise.rms_norm(x, self.weight, self.eps)
    dtype = x.dtype
    return self.norm(x.float()).to(dtype) * self.weight


def SelfAttention_forward(self, x, freqs, attn_mask: Optional[torch.Tensor] = None):
    q = run_norm(self.norm_q, run_projection(self.q, x))
    k = run_norm(self.norm_k, run_projection(self.k, x))
    v = run_projection(self.v, x)
    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)
    x = self.attn(q, k, v, attn_mask=attn_mask)
    return run_projection(self.o, x)


def CrossAttention_forward(
    self,
    x: torch.Tensor,
    y: torch.Tensor,
    ctx_mask: Optional[torch.Tensor] = None,
    *,
    linear_bias2d: bool = False,
):
    if self.has_image_input:
        img = y[:, :257]
        ctx = y[:, 257:]
        if ctx_mask is not None:
            ctx_mask = ctx_mask[..., 257:]
    else:
        ctx = y
    q = run_norm(
        self.norm_q,
        video_linear(self.q, x) if linear_bias2d else run_projection(self.q, x),
    )
    k = run_norm(self.norm_k, run_projection(self.k, ctx))
    v = run_projection(self.v, ctx)
    x = run_attention(
        self.attn, q, k, v, attn_mask=ctx_mask, label="video_cross_attention"
    )
    if self.has_image_input:
        k_img = run_norm(self.norm_k_img, run_projection(self.k_img, img))
        v_img = run_projection(self.v_img, img)
        y = run_attention(
            flash_attention,
            q,
            k_img,
            v_img,
            num_heads=self.num_heads,
            label="video_cross_attention",
        )
        x = x + y
    return video_linear(self.o, x) if linear_bias2d else run_projection(self.o, x)


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

flash_attention_wrapper = _port_wrapper(
    flash_attention,
    "openwam.model.video_backbone.wan.models.dit.flash_attention",
    "openwam.compute",
)

RMSNorm_forward_wrapper = _port_wrapper(
    RMSNorm_forward,
    "openwam.model.video_backbone.wan.models.dit.RMSNorm.forward",
    "openwam.compute",
)

SelfAttention_forward_wrapper = _port_wrapper(
    SelfAttention_forward,
    "openwam.model.video_backbone.wan.models.dit.SelfAttention.forward",
    "openwam.compute",
)

CrossAttention_forward_wrapper = _port_wrapper(
    CrossAttention_forward,
    "openwam.model.video_backbone.wan.models.dit.CrossAttention.forward",
    "openwam.compute",
)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    from turbo_physai.optimizations.models.openwam import enabled, pointwise
    from turbo_physai.optimizations.models.openwam.sac import extras_enabled

    if enabled("pointwise_compile") or extras_enabled():
        return pointwise.modulate(x, shift, scale)
    return x * (1 + scale) + shift


modulate_wrapper = _port_wrapper(
    modulate, "openwam.model.video_backbone.wan.models.dit.modulate", "openwam.compute"
)


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    from turbo_physai.optimizations.models.openwam import enabled, pointwise

    if enabled("rope_real"):
        return pointwise.rotate(x, freqs).flatten(2)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


rope_apply_wrapper = _port_wrapper(
    rope_apply,
    "openwam.model.video_backbone.wan.models.dit.rope_apply",
    "openwam.compute",
)


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex128
    return freqs_cis


precompute_freqs_cis_wrapper = _port_wrapper(
    precompute_freqs_cis,
    "openwam.model.video_backbone.wan.models.dit.precompute_freqs_cis",
    "openwam.compute",
)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


precompute_freqs_cis_3d_wrapper = _port_wrapper(
    precompute_freqs_cis_3d,
    "openwam.model.video_backbone.wan.models.dit.precompute_freqs_cis_3d",
    "openwam.compute",
)


def GateModule_forward(self, x, gate, residual):
    from turbo_physai.optimizations.models.openwam import enabled, pointwise
    from turbo_physai.optimizations.models.openwam.sac import extras_enabled

    if enabled("pointwise_compile") or extras_enabled():
        return pointwise.gate(x, gate, residual)
    return x + gate * residual


GateModule_forward_wrapper = _port_wrapper(
    GateModule_forward,
    "openwam.model.video_backbone.wan.models.dit.GateModule.forward",
    "openwam.compute",
)


def DiTBlock_init(
    self,
    has_image_input: bool,
    dim: int,
    num_heads: int,
    ffn_dim: int,
    eps: float = 1e-6,
):
    torch.nn.Module.__init__(self)
    self.dim = dim
    self.num_heads = num_heads
    self.ffn_dim = ffn_dim

    self.self_attn = SelfAttention(dim, num_heads, eps)
    self.cross_attn = CrossAttention(
        dim, num_heads, eps, has_image_input=has_image_input
    )
    from ..norms import LayerNorm

    norm = LayerNorm if enabled("lightop_norm") else nn.LayerNorm

    self.norm1 = norm(dim, eps=eps, elementwise_affine=False)
    self.norm2 = norm(dim, eps=eps, elementwise_affine=False)
    self.norm3 = norm(dim, eps=eps)
    self.ffn = nn.Sequential(
        nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
    )
    self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    self.gate = GateModule()


dit_block_init_wrapper = _port_wrapper(
    DiTBlock_init,
    "openwam.model.video_backbone.wan.models.dit.DiTBlock.__init__",
    "openwam.compute",
)


def fused_rope_ok(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> bool:
    if not (
        _HAVE_TRITON
        and os.environ.get("OPENWAM_FUSED_ROPE", "0") == "1"
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.dim() == 3
        and x.is_contiguous()
        and freqs.dtype == torch.complex128
        and freqs.device == x.device
    ):
        return False
    if freqs.dim() == 3:
        return (
            freqs.shape[0] == x.shape[1]
            and freqs.shape[1] == 1
            and x.shape[2] == num_heads * 2 * freqs.shape[2]
        )
    if freqs.dim() == 2:
        return (
            freqs.shape[0] == x.shape[1]
            and x.shape[2] == num_heads * 2 * freqs.shape[1]
        )
    return False


def rope_apply_auto(
    x: torch.Tensor, freqs: torch.Tensor, num_heads: int, reference_fn
) -> torch.Tensor:
    """``reference_fn(x, freqs, num_heads)`` (``rope_apply``) through the fused kernel
    when its guards hold. The first call also runs the reference and compares bitwise; on a
    mismatch the fused path is disabled for the process (logged)."""
    ref_freqs = freqs.unsqueeze(1) if freqs.dim() == 2 else freqs
    if not (_STATE["enabled"] and fused_rope_ok(x, freqs, num_heads)):
        return reference_fn(x, ref_freqs, num_heads)
    if not _STATE["checked"]:
        _STATE["checked"] = True
        with torch.no_grad():
            ref = reference_fn(x.detach(), ref_freqs, num_heads)
            out = fused_rope_apply(x.detach(), freqs, num_heads)
            same = torch.equal(ref.view(torch.int16), out.view(torch.int16))
        if not same:
            _STATE["enabled"] = False
            import logging

            logging.getLogger(__name__).warning(
                "fused RoPE differs from rope_apply on first call; disabled"
            )
            return reference_fn(x, ref_freqs, num_heads)
    return fused_rope_apply(x, freqs, num_heads)


rope_apply_auto_wrapper = _port_wrapper(
    rope_apply_auto,
    "openwam.model.kernels.rope_triton.rope_apply_auto",
    "openwam.compute",
)
