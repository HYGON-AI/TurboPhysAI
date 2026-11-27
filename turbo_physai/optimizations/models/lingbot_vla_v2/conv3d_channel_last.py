# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""patch embedding 的 3D 卷积使用 channels_last_3d，backward 的权重梯度使用 NCDHW。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.compiler.disable
def _conv3d_channels_last(hidden_states, weight, bias, stride):
    return F.conv3d(
        hidden_states.to(memory_format=torch.channels_last_3d),
        weight.to(memory_format=torch.channels_last_3d),
        bias,
        stride=stride,
    )


class _ChannelLastConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, weight, bias, stride):
        ctx.save_for_backward(hidden_states, weight)
        ctx.stride = stride
        ctx.has_bias = bias is not None
        return _conv3d_channels_last(hidden_states, weight, bias, stride)

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            raise RuntimeError(
                "lingbot-vla-v2.conv3d-channel-last: patch embedding input never requires grad"
            )
        grad_weight = torch.nn.grad.conv3d_weight(
            hidden_states.contiguous(),
            weight.shape,
            grad_output.contiguous(),
            stride=ctx.stride,
        )
        grad_bias = grad_output.sum(dim=(0, 2, 3, 4)) if ctx.has_bias else None
        return None, grad_weight, grad_bias, None


def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = hidden_states.view(
        -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
    )
    output = _ChannelLastConv3d.apply(
        hidden_states.to(dtype=self.proj.weight.dtype),
        self.proj.weight,
        self.proj.bias,
        self.proj.stride,
    )
    return output.reshape(-1, self.embed_dim)
