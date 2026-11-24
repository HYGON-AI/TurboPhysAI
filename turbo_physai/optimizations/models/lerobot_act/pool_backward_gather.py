# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""ACT max-pool backward gather, adapted from the v6 HCU candidate.

The ACT backbone pool is marked through an Engine-declared target.
Unsupported ACT pool inputs raise an error; other pools use the original implementation.
"""

import functools

import torch
import triton
import triton.language as tl


_MARKER = '_turbo_lerobot_act_pool_gather'


@triton.jit
def _pool_backward_gather(indices, grad, output, N: tl.constexpr, C: tl.constexpr,
                          H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = position < N * C * H * W
    channel = position % C
    column = (position // C) % W
    row = (position // (C * W)) % H
    batch = position // (C * W * H)
    OH: tl.constexpr = (H + 1) // 2
    OW: tl.constexpr = (W + 1) // 2
    result = tl.full((BLOCK,), 0.0, tl.float32)
    for dy in tl.static_range(2):
        for dx in tl.static_range(2):
            out_y = row // 2 + dy
            out_x = column // 2 + dx
            active = valid & (out_y < OH) & (out_x < OW)
            active = active & ((dy == 0) | (row % 2 == 1))
            active = active & ((dx == 0) | (column % 2 == 1))
            offset = ((batch * OH + out_y) * OW + out_x) * C + channel
            selected = tl.load(indices + offset, mask=active, other=-1) == row * W + column
            value = tl.load(grad + offset, mask=active & selected, other=0.0)
            result += value
    tl.store(output + position, result, mask=valid)


class _PoolGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        output, indices = torch.ops.aten.max_pool2d_with_indices.default(
            value, [3, 3], [2, 2], [1, 1], [1, 1], False
        )
        ctx.save_for_backward(value, indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        value, indices = ctx.saved_tensors
        if torch.is_grad_enabled():
            raise RuntimeError('ACT pool gather does not support higher-order gradients.')
        n, c, h, w = value.shape
        indices = indices.contiguous(memory_format=torch.channels_last)
        grad_output = grad_output.contiguous(memory_format=torch.channels_last)
        output = torch.empty_like(value, memory_format=torch.channels_last)
        _pool_backward_gather[(triton.cdiv(value.numel(), 256),)](
            indices, grad_output, output, N=n, C=c, H=h, W=w, BLOCK=256, num_warps=4
        )
        return output


def mark_act_pool(act):
    """Mark only the pool belonging to this ACT backbone, if present."""
    backbone = getattr(act, 'backbone', None)
    pool = getattr(backbone, 'maxpool', None)
    if not isinstance(pool, torch.nn.MaxPool2d):
        raise RuntimeError('ACT backbone maxpool must be a torch.nn.MaxPool2d.')
    setattr(pool, _MARKER, True)


def max_pool_backward_gather_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, value):
        if not getattr(self, _MARKER, False):
            return original(self, value)
        checks = (
            (value.ndim == 4, 'input must be four-dimensional'),
            (value.ndim > 0 and value.shape[0] > 0, 'batch must be nonempty'),
            (value.numel() < 2**31, 'input must have fewer than 2**31 elements'),
            (value.is_cuda, 'device must be HCU/CUDA'),
            (value.is_contiguous(memory_format=torch.channels_last), 'input must be channels_last'),
        )
        failures = [reason for valid, reason in checks if not valid]
        if failures:
            raise RuntimeError('ACT pool gather unsupported: ' + '; '.join(failures))
        return _PoolGather.apply(value)

    return wrapped
