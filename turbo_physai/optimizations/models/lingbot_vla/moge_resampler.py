# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""moge neck 中 Resampler(bilinear) 的融合优化。

官方实现（moge/model/modules.py）::

    nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in, out, 3, stride=1, padding=1, padding_mode='replicate'),
    )

在 inductor 下，upsample 会被拆成「NHWC 读 + NCHW 写」的逐元素 kernel（读 stride= C，非 coalesced），带宽利用率低。

这里把 upsample + replicate_pad 融合成一个 kernel。
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn as nn
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _up_pad_2x_kernel(x, y, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                      OW: tl.constexpr, OH: tl.constexpr, ROW: tl.constexpr,
                      OUT_FP16: tl.constexpr):
    row = tl.program_id(0)
    n = row // OH
    oh = row % OH
    cc = tl.arange(0, C)
    iw = tl.arange(0, W)[:, None]

    uy = tl.minimum(tl.maximum(oh - 1, 0), 2 * H - 1)
    fy = tl.maximum(uy.to(tl.float32) * 0.5 - 0.25, 0.0)
    iy0 = fy.to(tl.int32)
    iy1 = tl.minimum(iy0 + 1, H - 1)
    wy = tl.minimum(fy - iy0.to(tl.float32), 1.0)
    wy1 = 1.0 - wy

    b = n * H * W * C
    o0 = iy0 * W
    o1 = iy1 * W
    dst = row * ROW

    ip = tl.maximum(iw - 1, 0)
    inx = tl.minimum(iw + 1, W - 1)
    p0 = tl.load(x + b + (o0 + ip) * C + cc).to(tl.float32)
    m0 = tl.load(x + b + (o0 + iw) * C + cc).to(tl.float32)
    q0 = tl.load(x + b + (o0 + inx) * C + cc).to(tl.float32)
    p1 = tl.load(x + b + (o1 + ip) * C + cc).to(tl.float32)
    m1 = tl.load(x + b + (o1 + iw) * C + cc).to(tl.float32)
    q1 = tl.load(x + b + (o1 + inx) * C + cc).to(tl.float32)

    # 和 aten 的 upsample_bilinear2d 一样，先 W 方向加权再 H 方向
    even = (p0 * 0.25 + m0 * 0.75) * wy1 + (p1 * 0.25 + m1 * 0.75) * wy
    odd = (m0 * 0.75 + q0 * 0.25) * wy1 + (m1 * 0.75 + q1 * 0.25) * wy
    if OUT_FP16:
        tl.store(y + dst + ((2 * iw + 1) * C + cc), even.to(tl.float16))
        tl.store(y + dst + ((2 * iw + 2) * C + cc), odd.to(tl.float16))
    else:
        tl.store(y + dst + ((2 * iw + 1) * C + cc), even)
        tl.store(y + dst + ((2 * iw + 2) * C + cc), odd)

    # replicate pad 的左右边界
    l0 = tl.load(x + b + o0 * C + cc).to(tl.float32)
    l1 = tl.load(x + b + o1 * C + cc).to(tl.float32)
    r0 = tl.load(x + b + (o0 + W - 1) * C + cc).to(tl.float32)
    r1 = tl.load(x + b + (o1 + W - 1) * C + cc).to(tl.float32)
    left = l0 * wy1 + l1 * wy
    right = r0 * wy1 + r1 * wy
    if OUT_FP16:
        tl.store(y + dst + cc, left.to(tl.float16))
        tl.store(y + dst + (OW - 1) * C + cc, right.to(tl.float16))
    else:
        tl.store(y + dst + cc, left)
        tl.store(y + dst + (OW - 1) * C + cc, right)


def up_pad_2x(x: torch.Tensor, out_fp16: bool = False) -> torch.Tensor:
    """[N,C,H,W] channels_last -> [N,C,2H+2,2W+2] channels_last。"""
    n, c, h, w = x.shape
    oh, ow = 2 * h + 2, 2 * w + 2
    out_fp16 = out_fp16 or x.dtype == torch.float16
    y = torch.empty_strided(
        (n, c, oh, ow), (c * oh * ow, 1, ow * c, c),
        dtype=torch.float16 if out_fp16 else torch.float32, device=x.device)
    _up_pad_2x_kernel[(n * oh,)](x, y, c, h, w, ow, oh, ow * c, out_fp16,
                                 num_warps=16)
    return y


if not hasattr(torch.ops.turbophysai, "up_pad_2x"):

    @torch.library.custom_op("turbophysai::up_pad_2x", mutates_args=())
    def _up_pad_2x_op(x: torch.Tensor, out_fp16: bool) -> torch.Tensor:
        return up_pad_2x(x, out_fp16)

    @_up_pad_2x_op.register_fake
    def _up_pad_2x_fake(x, out_fp16):
        n, c, h, w = x.shape
        oh, ow = 2 * h + 2, 2 * w + 2
        dtype = torch.float16 if (out_fp16 or x.dtype == torch.float16) else torch.float32
        return torch.empty_strided((n, c, oh, ow), (c * oh * ow, 1, ow * c, c),
                                   dtype=dtype, device=x.device)


def _pow2(value: int) -> bool:
    """kernel 里用 tl.arange(0, C) / tl.arange(0, W)，要求长度是 2 的幂。"""

    return value > 0 and value & (value - 1) == 0


class FusedUpPad2x(nn.Module):
    """替代 ``nn.Upsample(scale_factor=2, mode='bilinear')`` + replicate pad。"""

    def __init__(self, out_fp16: bool = False):
        super().__init__()
        self.out_fp16 = out_fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.dim() == 4
            and x.dtype in (torch.float32, torch.float16)
            and _pow2(x.shape[1])
            and _pow2(x.shape[3])
        ):
            if not x.is_contiguous(memory_format=torch.channels_last):
                x = x.contiguous(memory_format=torch.channels_last)
            return torch.ops.turbophysai.up_pad_2x(x, self.out_fp16)
        # kernel 用 tl.arange(0, C) / tl.arange(0, W)，C 或 W 不是 2 的幂时会报错，这里回退到 aten 的 interpolate + pad。
        y = nn.functional.interpolate(x, scale_factor=2, mode="bilinear",
                                      align_corners=False)
        y = nn.functional.pad(y, (1, 1, 1, 1), mode="replicate")
        return y.to(torch.float16) if self.out_fp16 else y


def wrap_resampler_init(original, options=None):
    """moge.model.modules.Resampler: bilinear + 2x 时换成融合 kernel。"""
    out_fp16 = bool((options or {}).get("out_fp16", False))

    @functools.wraps(original)
    def wrapped(self, in_channels, out_channels, type_, scale_factor=2):
        original(self, in_channels, out_channels, type_, scale_factor)
        if type_ != "bilinear" or scale_factor != 2:
            return
        conv = self[1]
        if not isinstance(conv, nn.Conv2d) or tuple(conv.padding) != (1, 1):
            return
        conv.padding = (0, 0)
        conv.padding_mode = "zeros"
        conv._reversed_padding_repeated_twice = [0, 0, 0, 0]
        self[0] = FusedUpPad2x(out_fp16)
        logger.info(
            "lingbot-vla: fused bilinear upsample+replicate pad (%d->%d, out_fp16=%s)",
            in_channels, out_channels, out_fp16,
        )

    return wrapped
