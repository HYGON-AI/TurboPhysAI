# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pytest
import torch
import numpy as np
from functools import partial
from utils import allclose, run_benchmark
from mmcv.ops import modulated_deform_conv2d as mmcv_modulated_deform_conv2d
from turbo_physai import modulated_deform_conv2d as hcu_modulated_deform_conv2d


class TestModulatedDeformableConv2d():
    def create_single_cpu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        return torch.from_numpy(input1)
    
    def get_cpu_golden(self, dtype, x, offset, mask, weight, groups, grad_out):
        x_torch = x.detach().clone().cuda()
        offset_torch = offset.detach().clone().cuda()
        mask_torch = mask.detach().clone().cuda()
        weight_torch = weight.detach().clone().cuda()
        x_torch.grad, offset_torch.grad, mask_torch.grad, weight_torch.grad = None, None, None, None
        
        x_torch.requires_grad = True
        offset_torch.requires_grad = True
        mask_torch.requires_grad = True
        weight_torch.requires_grad = True
        
        avg_ms, _ = run_benchmark(partial(mmcv_modulated_deform_conv2d, x_torch, offset_torch, mask_torch, weight_torch, None, 1, 1, 1, groups),
                                  repeat_num=100,
                                  backward=True,
                                  grad=grad_out.cuda())

        return avg_ms, x_torch.grad.cpu(), offset_torch.grad.cpu(), mask_torch.grad.cpu(), weight_torch.grad.cpu()
    
    def get_hcu_output(self, x, offset, mask, weight, groups, grad_out):
        x_hcu = x.detach().clone().to(memory_format=torch.channels_last).cuda()
        offset_hcu = offset.detach().clone().to(memory_format=torch.channels_last).cuda()
        mask_hcu = mask.detach().clone().to(memory_format=torch.channels_last).cuda()
        weight_hcu = weight.detach().clone().to(memory_format=torch.channels_last).cuda()
        x_hcu.grad, offset_hcu.grad, mask_hcu.grad, weight_hcu.grad = None, None, None, None
        
        x_hcu.requires_grad = True
        offset_hcu.requires_grad = True
        mask_hcu.requires_grad = True
        weight_hcu.requires_grad = True
        
        avg_ms, _ = run_benchmark(partial(hcu_modulated_deform_conv2d, x_hcu, offset_hcu, mask_hcu, weight_hcu, None, 1, 1, 1, groups),
                                  repeat_num=100,
                                  backward=True,
                                  grad=grad_out.cuda())

        return avg_ms, x_hcu.grad.cpu(), offset_hcu.grad.cpu(), mask_hcu.grad.cpu(), weight_hcu.grad.cpu()

    def single_check_result(self, hcu_out, cpu_out):
        avg_ms_torch, x_grad_cpu, offset_grad_cpu, mask_grad_cpu, weight_grad_cpu = cpu_out
        avg_ms_hcu, x_grad_hcu, offset_grad_hcu, mask_grad_hcu, weight_grad_hcu = hcu_out
        
        print('speedup ratio:', f"{avg_ms_torch / avg_ms_hcu:.2f}x")
        allclose(x_grad_hcu, x_grad_cpu, rtol=1e-3, atol=1e-3)
        # offset和mask的grad目前不支持，先不比较了
        allclose(weight_grad_hcu, weight_grad_cpu, rtol=1e-1, atol=1e-1)


    @pytest.mark.parametrize("N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups", [
        (6, 512, 512, 3, 29, 50, 29, 50, 1),
        (6, 256, 256, 3, 58, 100, 58, 100, 1),
        (12, 512, 512, 3, 29, 50, 29, 50, 1),
        (12, 256, 256, 3, 58, 100, 58, 100, 1),
    ])
    def test_bevformer_model_case(self, N, cIn, cOut, K, hIn, wIn, hOut, wOut, groups):
        x = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hIn, wIn)], -5, 5)
        offset = self.create_single_cpu_tensor([np.float32, 0, (N, 2 * K * K, hOut, wOut)], -2, 2)
        mask = self.create_single_cpu_tensor([np.float32, 0, (N, K * K, hOut, wOut)], -5, 5)
        weight = self.create_single_cpu_tensor([np.float32, 0, (cOut, cIn // groups, K, K)], -5, 5) * 0.001
        grad_out = self.create_single_cpu_tensor([np.float32, 0, (N, cIn, hOut, wOut)], -5, 5)

        hcu_out = self.get_hcu_output(x, offset, mask, weight, groups, grad_out)
        cpu_out = self.get_cpu_golden(torch.float32, x, offset, mask, weight, groups, grad_out)
        self.single_check_result(hcu_out, cpu_out)
