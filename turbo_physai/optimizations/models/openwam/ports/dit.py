# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import openwam.model.video_backbone.wan.models.dit as _source
import torch.nn as nn
from openwam.model.video_backbone.wan.models.dit import (
    CrossAttention,
    F,
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    GateModule,
    SAGE_ATTN_AVAILABLE,
    SelfAttention,
    rearrange,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openwam.model.video_backbone.wan.models.dit import (
        Optional,
    )
from turbo_physai.optimizations.models.openwam import enabled, pointwise
from turbo_physai.optimizations.models.openwam.attention import fa2_padding
from turbo_physai.optimizations.models.openwam.linear_bias2d import video_linear
from turbo_physai.optimizations.models.openwam.norms import LayerNorm
from turbo_physai.optimizations.models.openwam.sac import (
    dispatcher,
    extras_enabled,
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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    if enabled("pointwise_compile") or extras_enabled():
        return pointwise.modulate(x, shift, scale)
    return x * (1 + scale) + shift


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    if enabled("rope_real"):
        return pointwise.rotate(x, freqs).flatten(2)
    x_out = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


def RMSNorm_forward(self, x):
    if enabled("pointwise_compile") or enabled("lightop_norm"):
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


def GateModule_forward(self, x, gate, residual):
    if enabled("pointwise_compile") or extras_enabled():
        return pointwise.gate(x, gate, residual)
    return x + gate * residual


def DiTBlock___init__(
    self,
    has_image_input: bool,
    dim: int,
    num_heads: int,
    ffn_dim: int,
    eps: float = 1e-6,
):
    super(_source.DiTBlock, self).__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.ffn_dim = ffn_dim

    self.self_attn = SelfAttention(dim, num_heads, eps)
    self.cross_attn = CrossAttention(
        dim, num_heads, eps, has_image_input=has_image_input
    )
    self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
    self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
    self.norm3 = LayerNorm(dim, eps=eps)
    self.ffn = nn.Sequential(
        nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
    )
    self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    self.gate = GateModule()


from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

flash_attention_wrapper = _port_wrapper(
    flash_attention,
    "openwam.model.video_backbone.wan.models.dit.flash_attention",
    "openwam.compute",
)

modulate_wrapper = _port_wrapper(
    modulate, "openwam.model.video_backbone.wan.models.dit.modulate", "openwam.compute"
)

rope_apply_wrapper = _port_wrapper(
    rope_apply,
    "openwam.model.video_backbone.wan.models.dit.rope_apply",
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

GateModule_forward_wrapper = _port_wrapper(
    GateModule_forward,
    "openwam.model.video_backbone.wan.models.dit.GateModule.forward",
    "openwam.compute",
)

DiTBlock___init___wrapper = _port_wrapper(
    DiTBlock___init__,
    "openwam.model.video_backbone.wan.models.dit.DiTBlock.__init__",
    "openwam.compute",
)
