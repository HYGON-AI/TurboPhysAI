# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pytest
import torch
import numpy as np
from functools import partial
from utils import allclose, run_benchmark
from turbo_physai import deformable_aggregation_function as hcu_deformable_aggregation_function
from deformable_aggregation_np import DeformableAggregationFunction as cpu_deformable_aggregation_function


def cpu_gen_inputs(B, C, numGroups, anchor, pts, num_feat, cam, scale, H, W):
    feature_maps = np.random.rand(B, num_feat, C).astype(np.float32)
    spatial_shape = torch.tensor([[[H, W]] * scale] * cam, dtype=torch.int32).numpy()
    scale_start_index = torch.zeros((cam, scale), dtype=torch.int32).numpy()
    feat_area_per_map = H * W
    for i in range(cam):
        for s in range(scale):
            idx = i * scale + s
            scale_start_index[i, s] = idx * feat_area_per_map
    sample_location = np.random.rand(B, anchor, pts, cam, 2).astype(np.float32)
    weights = np.random.rand(B, anchor, pts, cam, scale, numGroups).astype(np.float32)

    return feature_maps, spatial_shape, scale_start_index, sample_location, weights


class TestDeformableAggregation():
    def get_cpu_golden(self, feature_maps, spatial_shape, scale_start_index, sample_location, weights):
        out = cpu_deformable_aggregation_function.golden_deformable_aggregation(feature_maps, spatial_shape, scale_start_index, sample_location, weights)
        grad_mc_ms_feat, grad_sampling_location, grad_weights = cpu_deformable_aggregation_function.golden_deformable_aggregation_grad(feature_maps, spatial_shape, scale_start_index, sample_location, weights)
        return torch.from_numpy(out), torch.from_numpy(grad_mc_ms_feat), torch.from_numpy(grad_sampling_location), torch.from_numpy(grad_weights)
    
    def get_hcu_output(self, feature_maps, spatial_shape, scale_start_index, sample_location, weights):
        torch_feature_maps = torch.from_numpy(feature_maps).cuda()
        torch_spatial_shape = torch.from_numpy(spatial_shape).cuda()
        torch_scale_start_index = torch.from_numpy(scale_start_index).cuda()
        torch_sample_location = torch.from_numpy(sample_location).cuda()
        torch_weights = torch.from_numpy(weights).cuda()
        torch_feature_maps.grad, torch_sample_location.grad, torch_weights.grad = None, None, None
        
        torch_feature_maps.requires_grad = True
        torch_sample_location.requires_grad = True
        torch_weights.requires_grad = True
        
        avg_ms, out = run_benchmark(partial(hcu_deformable_aggregation_function, torch_feature_maps, torch_spatial_shape, torch_scale_start_index, torch_sample_location, torch_weights),
                                  repeat_num=100,
                                  backward=True,
                                  grad=torch.ones([torch_feature_maps.shape[0], torch_sample_location.shape[1], torch_feature_maps.shape[2]], dtype=torch.float32).cuda())

        return avg_ms, out.cpu(), torch_feature_maps.grad.cpu(), torch_sample_location.grad.cpu(), torch_weights.grad.cpu()

    def single_check_result(self, hcu_out, cpu_out):
        out_cpu, feature_maps_grad_cpu, sample_location_grad_cpu, weights_grad_cpu = cpu_out
        avg_ms_hcu, out_hcu, feature_maps_grad_hcu, sample_location_grad_hcu, weights_grad_hcu = hcu_out
        
        print(f"hcu_deformable_aggregation_function cost time: {avg_ms_hcu:.2f} ms")
        allclose(out_hcu, out_cpu, rtol=1e-3, atol=1e-3)
        allclose(feature_maps_grad_hcu, feature_maps_grad_cpu.reshape(feature_maps_grad_hcu.shape), rtol=1e-3, atol=1e-3)
        allclose(sample_location_grad_hcu, sample_location_grad_cpu.reshape(sample_location_grad_hcu.shape), rtol=1e-3, atol=1e-3)
        allclose(weights_grad_hcu, weights_grad_cpu.reshape(weights_grad_hcu.shape), rtol=1e-3, atol=1e-3)


    @pytest.mark.parametrize("B, C, numGroups, anchor, pts, num_feat, cam, scale, H, W", [
        #(6, 256, 8, 1220, 13, 89760, 6, 4, 44, 85), # Sparse4D 模型中的size 因数据量太大，numpy需跑一天，注释掉
        (1, 32, 8, 10, 10, 2816, 1, 1, 32, 88),
        (1, 64, 16, 13, 50, 2816, 1, 1, 32, 88),
        (5, 64, 16, 13, 50, 2816, 1, 1, 32, 88),
        (5, 64, 16, 13, 31, 2816, 1, 1, 32, 88),
        (10, 64, 16, 18, 50, 2816, 1, 1, 32, 88),
        (10, 32, 8, 10, 10, 2816, 1, 1, 32, 88),
    ])
    def test_sparse4d_model_case(self, B, C, numGroups, anchor, pts, num_feat, cam, scale, H, W):
        feature_maps, spatial_shape, scale_start_index, sample_location, weights = cpu_gen_inputs(B, C, numGroups, anchor, pts, num_feat, cam, scale, H, W)

        hcu_out = self.get_hcu_output(feature_maps, spatial_shape, scale_start_index, sample_location, weights)
        cpu_out = self.get_cpu_golden(feature_maps, spatial_shape, scale_start_index, sample_location, weights)
        self.single_check_result(hcu_out, cpu_out)
