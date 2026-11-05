# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from einops import rearrange
from openwam.model.video_backbone.wan.models.dit import (
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    rope_apply as _rope_apply,
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
from turbo_physai.optimizations.models.openwam.norms import lightop_rms_norm
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
    if enabled("lightop_norm"):
        fast = lightop_rms_norm(x, self.weight, self.eps)
        if fast is not None:
            return fast
    dtype = x.dtype
    return self.norm(x.float()).to(dtype) * self.weight


def SelfAttention_forward(self, x, freqs, attn_mask: Optional[torch.Tensor] = None):
    q = run_norm(self.norm_q, run_projection(self.q, x))
    k = run_norm(self.norm_k, run_projection(self.k, x))
    v = run_projection(self.v, x)
    q = _rope_apply(q, freqs, self.num_heads)
    k = _rope_apply(k, freqs, self.num_heads)
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
