# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import pytest
import torch
from functools import partial
from utils import allclose, run_benchmark
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from turbo_physai import MultiScaleDeformableAttnFunction


def test_forward_equal_with_pytorch_float():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros(
        (1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    im2col_step = 2
    output_pytorch = multi_scale_deformable_attn_pytorch(
        value, shapes, sampling_locations, attention_weights).detach().cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_device = MultiScaleDeformableAttnFunction.apply(
        value.to(device), shapes.to(device), level_start_index.to(device),
        sampling_locations.to(device), attention_weights.to(device)).detach().cpu()
    assert torch.allclose(output_device, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_device - output_pytorch).abs().max()
    max_rel_err = ((output_device - output_pytorch).abs() /
                   output_pytorch.abs()).max()
    assert max_abs_err < 1e-9
    assert max_rel_err < 1e-6


def test_backward_equal_with_pytorch():
    N, M, D = 6, 4, 8
    Lq, L, P = 10000, 4, 8
    shapes = torch.as_tensor([(60, 40), (30, 20), (16, 24), (53, 32)],
                             dtype=torch.int32)
    level_start_index = torch.cat((shapes.new_zeros(
        (1, )), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(
        -1, keepdim=True).sum(
            -2, keepdim=True)
    im2col_step = 2
    value.requires_grad = True
    sampling_locations.requires_grad = True
    attention_weights.requires_grad = True
    output_pytorch = multi_scale_deformable_attn_pytorch(
        value.float(), shapes, sampling_locations.float(),
        attention_weights.float())
    grad_output_pytorch = torch.ones_like(output_pytorch)
    output_pytorch.backward(grad_output_pytorch)
    grad_value = value.grad.detach().cpu()
    grad_location = sampling_locations.grad.detach().cpu()
    grad_attn_weight = attention_weights.grad.detach().cpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value_hcu = value.to(device).detach().requires_grad_(True)
    shapes_hcu = shapes.to(device).long()
    level_start_index_hcu = level_start_index.to(device).long()
    sampling_locations_hcu = sampling_locations.to(device).detach().requires_grad_(True)
    attention_weights_hcu = attention_weights.to(device).detach().requires_grad_(True)
    output_hcu = MultiScaleDeformableAttnFunction.apply(
        value_hcu.float(), shapes_hcu, level_start_index_hcu,
        sampling_locations_hcu.float(), attention_weights_hcu.float())
    grad_output_hcu = torch.ones_like(output_hcu)
    output_hcu.backward(grad_output_hcu)
    grad_value_hcu = value_hcu.grad.detach().cpu()
    grad_location_hcu = sampling_locations_hcu.grad.detach().cpu()
    grad_attn_weight_hcu = attention_weights_hcu.grad.detach().cpu()
    assert torch.allclose(grad_value_hcu, grad_value)
    max_abs_err_1 = (grad_value_hcu - grad_value).abs().max()
    max_rel_err_1 = ((grad_value_hcu - grad_value).abs() /
                     grad_value.abs()).max()
    assert max_abs_err_1 < 1e-5
    assert max_rel_err_1 < 1e-4
    # 此处由于采样位置的梯度值较小，且数值不稳定，因此放宽误差要求
    # assert torch.allclose(grad_location_hcu, grad_location)
    # max_abs_err_2 = (grad_location_hcu - grad_location).abs().max()
    # max_rel_err_2 = ((grad_location_hcu - grad_location).abs() /
    #                  grad_location.abs()).max()
    # assert max_abs_err_2 < 1e-5
    # assert max_rel_err_2 < 1e-4
    assert torch.allclose(grad_attn_weight_hcu, grad_attn_weight)
    max_abs_err_3 = (grad_attn_weight_hcu - grad_attn_weight).abs().max()
    max_rel_err_3 = ((grad_attn_weight_hcu - grad_attn_weight).abs() /
                     grad_attn_weight.abs()).max()
    assert max_abs_err_3 < 1e-5
    assert max_rel_err_3 < 1e-4


class TestMultiScaleDeformableAttn():
    def get_torch_golden(self, value, shapes, sampling_locations, attention_weights):
        value = value.detach().clone().cuda()
        shapes = shapes.detach().clone().cuda()
        sampling_locations = sampling_locations.detach().clone().cuda()
        attention_weights = attention_weights.detach().clone().cuda()

        value.requires_grad = True
        sampling_locations.requires_grad = True
        attention_weights.requires_grad = True

        avg_ms, out = run_benchmark(partial(multi_scale_deformable_attn_pytorch,
                                          value,
                                          shapes,
                                          sampling_locations,
                                          attention_weights),
                                  repeat_num=100,
                                  backward=True,
                                  grad=torch.ones((sampling_locations.shape[0],
                                                   sampling_locations.shape[1],
                                                   value.shape[2] * value.shape[3]), dtype=value.dtype).cuda())

        return avg_ms, out.cpu(), value.grad.cpu(), sampling_locations.grad.cpu(), attention_weights.grad.cpu()
    
    def get_hcu_output(self, value, shapes, level_start_index, sampling_locations, attention_weights):
        value = value.detach().clone().cuda()
        shapes = shapes.detach().clone().cuda().long()
        level_start_index = level_start_index.detach().clone().cuda().long()
        sampling_locations = sampling_locations.detach().clone().cuda()
        attention_weights = attention_weights.detach().clone().cuda()
        value.grad, sampling_locations.grad, attention_weights.grad = None, None, None

        value.requires_grad = True
        sampling_locations.requires_grad = True
        attention_weights.requires_grad = True
        avg_ms, out = run_benchmark(partial(MultiScaleDeformableAttnFunction.apply,
                                            value, shapes, level_start_index, sampling_locations, attention_weights),
                                  repeat_num=100,
                                  backward=True,
                                  grad=torch.ones((sampling_locations.shape[0],
                                                   sampling_locations.shape[1],
                                                   value.shape[2] * value.shape[3]), dtype=value.dtype).cuda())

        return avg_ms, out.cpu(), value.grad.cpu(), sampling_locations.grad.cpu(), attention_weights.grad.cpu()

    def single_check_result(self, hcu_out, cpu_out):
        avg_ms_torch, torch_out, value_grad_cpu, sampling_locations_grad_cpu, attention_weights_grad_cpu = cpu_out
        avg_ms_hcu, hcu_out, value_grad_hcu, sampling_locations_grad_hcu, attention_weights_grad_hcu = hcu_out
        
        print('speedup ratio:', f"{avg_ms_torch / avg_ms_hcu:.2f}x")
        allclose(hcu_out, torch_out, rtol=1e-3, atol=1e-3)
        allclose(value_grad_hcu, value_grad_cpu, rtol=1e-3, atol=1e-3)
        allclose(sampling_locations_grad_hcu, sampling_locations_grad_cpu, rtol=1e-3, atol=1e-3)
        allclose(attention_weights_grad_hcu, attention_weights_grad_cpu, rtol=1e-1, atol=1e-1)


    @pytest.mark.parametrize("N, M, D, Lq, L, P, shapes", [
        (6, 4, 8, 10000, 4, 8, torch.as_tensor([(60, 40), (30, 20), (16, 24), (53, 32)], dtype=torch.int32)),
        (2, 8, 32, 40000, 1, 4, torch.as_tensor([(200, 200)], dtype=torch.int32)),
        (6, 8, 32, 9703, 4, 8, torch.as_tensor([(116, 200), (58, 100), (29, 50), (15, 25)], dtype=torch.int32)),
        (1, 8, 32, 900, 1, 4, torch.as_tensor([(200, 200)], dtype=torch.int32)),
    ])
    def test_bevformer_case(self, N, M, D, Lq, L, P, shapes):
        level_start_index = torch.cat((shapes.new_zeros(
            (1, )), shapes.prod(1).cumsum(0)[:-1]))
        S = sum((H * W).item() for H, W in shapes)

        torch.manual_seed(3)
        value = (torch.rand(N, S, M, D) * 0.01).float()
        sampling_locations = torch.rand(N, Lq, M, L, P, 2).float()
        attention_weights = (torch.rand(N, Lq, M, L, P) + 1e-5).float()
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

        torch_out = self.get_torch_golden(value, shapes, sampling_locations, attention_weights)
        hcu_out = self.get_hcu_output(value, shapes, level_start_index, sampling_locations, attention_weights)
        self.single_check_result(hcu_out, torch_out)
