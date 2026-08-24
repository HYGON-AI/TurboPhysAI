# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pytest
import torch
import numpy as np
from functools import partial
from utils import allclose, run_benchmark
from torch.nn.functional import interpolate as torch_interpolate
from turbo_physai import interpolate as hcu_interpolate


class TestUpSampleBilinear2d():
    def create_single_cpu_tensor(self, item, minvalue, maxvalue):
        dtype = item[0]
        format1 = item[1]
        shape = item[2]
        input1 = np.random.uniform(minvalue, maxvalue, shape).astype(dtype)
        return torch.from_numpy(input1)
    
    def get_cpu_golden(self, input, size, scale_factor, mode, align_corners, grad_out):
        input_torch = input.detach().clone().cuda()
        input_torch.grad = None
        
        input_torch.requires_grad = True
        
        avg_ms, out = run_benchmark(partial(torch_interpolate, input_torch, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
                                  repeat_num=100,
                                  backward=True,
                                  grad=grad_out.cuda())

        return avg_ms, out.cpu(), input_torch.grad.cpu()
    
    def get_hcu_output(self, input, size, scale_factor, mode, align_corners, grad_out):
        input_hcu = input.detach().clone().cuda()
        input_hcu.grad = None
        
        input_hcu.requires_grad = True
        
        avg_ms, out = run_benchmark(partial(hcu_interpolate, input_hcu, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
                                  repeat_num=100,
                                  backward=True,
                                  grad=grad_out.cuda())

        return avg_ms, out.cpu(), input_hcu.grad.cpu()

    def single_check_result(self, hcu_out, cpu_out):
        avg_ms_torch, out_torch, x_grad_cpu = cpu_out
        avg_ms_hcu, out_hcu, x_grad_hcu = hcu_out

        print('speedup ratio:', f"{avg_ms_torch / avg_ms_hcu:.2f}x")
        rtol=1e-3
        atol=1e-3
        if out_hcu.dtype in [torch.float16, torch.bfloat16]:
            rtol = 1e-2
            atol = 1e-2
        allclose(out_hcu, out_torch, rtol=rtol, atol=atol)
        allclose(x_grad_hcu, x_grad_cpu, rtol=rtol, atol=atol)


    @pytest.mark.parametrize("N, C, h_in , w_in, size, scale_factor, mode, align_corners, dtype", [
        (6, 256, 128, 248, None, 2.0, 'bilinear', False, np.float32),
        (6, 256, 128, 248, None, 2.0, 'bilinear', False, np.float16),
        (48, 384, 16, 32, None, 2.0, 'bilinear', False, np.float32),
        (6, 512, 16, 31, None, 4.0, 'bilinear', False, np.float16),
        (6, 512, 64, 124, None, 2.0, 'bilinear', False, np.float16),
        (48, 64, 128, 256, None, 2.0, 'bilinear', False, np.float32),
    ])
    def test_BOSHI_model_upsample_case(self, N, C, h_in, w_in, size, scale_factor, mode, align_corners, dtype):
        input = self.create_single_cpu_tensor([dtype, 0, (N, C, h_in, w_in)], 0, 1)
        if size is not None:
            if isinstance(size, (list, tuple)):
                output_size = size
            else:
                output_size = [size for _ in range(2)]
        else:
            if isinstance(scale_factor, (list, tuple)):
                output_size = [int(h_in * scale_factor[0]), int(w_in * scale_factor[1])]
            else:
                output_size = [int(h_in * scale_factor), int(w_in * scale_factor)]
        grad_out = self.create_single_cpu_tensor([dtype, 0, (N, C, output_size[0], output_size[1])], 0, 1)
        
        hcu_out = self.get_hcu_output(input, size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, grad_out=grad_out)
        cpu_out = self.get_cpu_golden(input, size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, grad_out=grad_out)
        self.single_check_result(hcu_out, cpu_out)
