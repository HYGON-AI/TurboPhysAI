# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from test.utils import allclose, run_benchmark
from turbo_physai import grid_sample

pytestmark = pytest.mark.hcu

torch.manual_seed(2026)

class GridSampleTorch(nn.Module):
    """使用 torch.nn.functional.grid_sample 的参考实现"""

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=False):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x, grid):
        return F.grid_sample(
            x, grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )

class GridSampleTurboPhysAI(nn.Module):

    def __init__(self, mode='bilinear', padding_mode='zeros', align_corners=False):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x, grid):
        return grid_sample(
            x, grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners
        )


def get_data(batchsize=48, channels=64, h_in=32, w_in=64, h_out=None, w_out=None, align_corners=False):
    """生成 input、grid、grad"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    x = torch.rand((batchsize, channels, h_in, w_in)).cuda()
    grad = torch.rand((batchsize, channels, h_out, w_out)).cuda()

    # 创建采样网格: (N, H_out, W_out, 2)，坐标范围 [-1, 1]
    y = torch.linspace(-1, 1, h_out).cuda()
    x_coord = torch.linspace(-1, 1, w_out).cuda()
    grid_y, grid_x = torch.meshgrid(y, x_coord, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = grid.expand(batchsize, -1, -1, -1)

    x.requires_grad_(True)
    grid.requires_grad_(True)

    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.clone().detach().requires_grad_(True)
    grid1 = grid.clone().detach().requires_grad_(True)
    grid2 = grid.clone().detach().requires_grad_(True)

    return x1, x2, grid1, grid2, grad

def get_nhwc_data(batchsize=48, channels=64, h_in=32, w_in=64, h_out=None, w_out=None, align_corners=False):
    """生成 input、grid、grad"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    x = torch.rand((batchsize, channels, h_in, w_in)).cuda()
    grad = torch.rand((batchsize, channels, h_out, w_out)).cuda()

    # 创建采样网格: (N, H_out, W_out, 2)，坐标范围 [-1, 1]
    y = torch.linspace(-1, 1, h_out).cuda()
    x_coord = torch.linspace(-1, 1, w_out).cuda()
    grid_y, grid_x = torch.meshgrid(y, x_coord, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    grid = grid.expand(batchsize, -1, -1, -1)

    x.requires_grad_(True)
    grid.requires_grad_(True)

    x1 = x.clone().detach().requires_grad_(True)
    x2 = x.to(memory_format=torch.channels_last).clone().detach().requires_grad_(True)
    grid1 = grid.clone().detach().requires_grad_(True)
    grid2 = grid.to(memory_format=torch.channels_last).clone().detach().requires_grad_(True)

    grad1 = grad.clone().detach()
    grad2 = grad.to(memory_format=torch.channels_last).clone().detach()

    return x1, x2, grid1, grid2, grad1, grad2

def get_model(mode='bilinear', padding_mode='zeros', align_corners=False):
    """返回 (torch 参考, lightop fuse)"""
    torch_grid_sample = GridSampleTorch(
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    ).cuda()
    fuse_grid_sample = GridSampleTurboPhysAI(
        mode=mode, padding_mode=padding_mode, align_corners=align_corners
    ).cuda()

    return torch_grid_sample, fuse_grid_sample

@pytest.mark.parametrize("batchsize, channels, h_in, w_in, h_out, w_out, repeat_num, mode, padding_mode, align_corners", [
    (48, 64, 32, 64, 128, 248, 100, 'bilinear', 'zeros', True),
])
def test_speed_forward(batchsize, channels, h_in, w_in, h_out, w_out,
                       repeat_num,
                       mode, padding_mode, align_corners):
    """仅测试前向：速度对比 + 输出正确性对比"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    x1, x2, grid1, grid2, grad = get_data(
        batchsize, channels, h_in, w_in, h_out, w_out, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    torch_time, y1 = run_benchmark(partial(f1, x1, grid1), repeat_num)
    das_time, y2 = run_benchmark(partial(f2, x2, grid2), repeat_num)

    # print('torch_grid_sample (forward)', 'average time:', torch_time, 'ms')
    # print('fuse_grid_sample (forward)', 'average time:', das_time, 'ms')
    print('speedup ratio:', f"{torch_time / das_time:.2f}x")

    allclose(y1, y2, rtol=1e-05, atol=1e-05)


@pytest.mark.parametrize("batchsize, channels, h_in, w_in, h_out, w_out, repeat_num, mode, padding_mode, align_corners", [
    (48, 64, 32, 64, 128, 248, 100, 'bilinear', 'zeros', True),
])
def test_speed(batchsize, channels, h_in, w_in, h_out, w_out,
               repeat_num,
               mode, padding_mode, align_corners):
    """仅测试反向：速度对比 + 梯度正确性对比"""
    h_out = h_out or h_in
    w_out = w_out or w_in

    x1, x2, grid1, grid2, grad = get_data(
        batchsize, channels, h_in, w_in, h_out, w_out, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    torch_time, y1 = run_benchmark(partial(f1, x1, grid1), repeat_num, backward=True, grad=grad)
    das_time, y2 = run_benchmark(partial(f2, x2, grid2), repeat_num, backward=True, grad=grad)

    # print('torch_grid_sample (backward)', 'average time:', torch_time, 'ms')
    # print('fuse_grid_sample (backward)', 'average time:', das_time, 'ms')
    print('speedup ratio:', f"{torch_time / das_time:.2f}x")

    allclose(y1, y2, rtol=1e-05, atol=1e-05)
    allclose(x1.grad, x2.grad, rtol=1e-05, atol=1e-05)
    allclose(grid1.grad, grid2.grad, rtol=1e-05, atol=1e-05)


@pytest.mark.parametrize("batchsize, channels, h_in, w_in, h_out, w_out, repeat_num, mode, padding_mode, align_corners", [
    (48, 64, 32, 64, 128, 248, 100, 'bilinear', 'zeros', True),
])
def test_nhwc_speed(batchsize, channels, h_in, w_in, h_out, w_out,
                    repeat_num, mode, padding_mode, align_corners):
    h_out = h_out or h_in
    w_out = w_out or w_in

    x1, x2, grid1, grid2, grad1, grad2 = get_nhwc_data(
        batchsize, channels, h_in, w_in, h_out, w_out, align_corners=align_corners
    )
    f1, f2 = get_model(mode=mode, padding_mode=padding_mode, align_corners=align_corners)

    torch_time, y1 = run_benchmark(partial(f1, x1, grid1), repeat_num, backward=True, grad=grad1)
    das_time, y2 = run_benchmark(partial(f2, x2, grid2), repeat_num, backward=True, grad=grad2)

    # print('torch_grid_sample', 'average time is', torch_time, 'ms')
    # print('fuse_grid_sample', 'average time is', das_time, 'ms')
    print('speedup ratio:', f"{torch_time / das_time:.2f}x")

    allclose(y1, y2, rtol=1e-05, atol=1e-05)
    allclose(x1.grad, x2.grad, rtol=1e-05, atol=1e-05)
    allclose(grid1.grad, grid2.grad, rtol=1e-05, atol=1e-03)
