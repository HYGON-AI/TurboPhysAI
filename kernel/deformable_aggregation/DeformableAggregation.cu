// Derived from Sparse4D: projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu
// Sparse4D commit 249ffbb695f4e9db628d953e2bf6d36de04bbb69.
// Copyright (c) 2024 Horizon Robotics
// SPDX-License-Identifier: MIT
// Copyright 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <THC/THCAtomics.cuh>

#include <iostream>
#include <stdlib.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

__device__ float bilinear_sampling(
    const float *&bottom_data, const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr
) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

struct float2_t{
  float a;
  float b;
};

__forceinline__ __device__
float2_t warp_reduce_sum(float2_t val, int max = 32) {
  for (int offset = max; offset > 0; offset >>= 1) {
    val.a += __shfl_down(val.a, offset);
    val.b += __shfl_down(val.b, offset);
  }
  return val;
}

template <int blocksize>
__forceinline__ __device__
float2_t block_reduce_sum(float2_t val, float2_t* shared) {
  const int lid = threadIdx.x % 64;
  const int wid = threadIdx.x / 64;
  constexpr int share_size = blocksize / 64;

  val = warp_reduce_sum(val);

  if constexpr (blocksize == 64) return val;

  if (lid == 0 && wid < share_size) {
    shared[wid] = val;
  }

  __syncthreads();

  if (wid == 0 && lid < share_size) {
    val = shared[lid];
    val = warp_reduce_sum(val, share_size / 2);
  }

  return val;

}

__device__ void bilinear_sampling_grad_sp(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights,
    float2_t* s_data) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;


  const int valid1 = (h_low >= 0 && w_low >= 0);
  const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
  float v1 = valid1 ? bottom_data[ptr1] : 0.0f;
  if (valid1) {
#ifdef __gfx936__
    __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
#endif
  }

  const int valid2 = (h_low >= 0 && w_high <= width - 1);
  const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
  float v2 = valid2 ? bottom_data[ptr2] : 0.0f;
  if (valid2) {
#ifdef __gfx936__
    __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
#endif
  }

  const int valid3 = (h_high <= height - 1 && w_low >= 0);
  const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
  float v3 = valid3 ? bottom_data[ptr3] : 0.0f;
  if (valid3) {
#ifdef __gfx936__
    __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
#endif
  }

  const int valid4 = (h_high <= height - 1 && w_high <= width - 1);
  const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
  float v4 = valid4 ? bottom_data[ptr4] : 0.0f;
  if (valid4) {
#ifdef __gfx936__
    __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
#endif
  }

  grad_h_weight += (-hw * v1) + (-lw * v2) + ( hw * v3) + ( lw * v4);
  grad_w_weight += (-hh * v1) + ( hh * v2) + (-lh * v3) + ( lh * v4);

  float2_t spl;
  spl.a = width * grad_w_weight * top_grad_mc_ms_feat;
  spl.b = height * grad_h_weight * top_grad_mc_ms_feat;

  spl = block_reduce_sum<256>(spl, s_data);

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  // TODO: 尝试 grad_weights 部分的规约, 区间设置
  //! | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  //! 区间不固定的情况怎么规约
  //! 还有线程号的对应关系
  //! ----------- 仅对特定参数优化----------------

  //! 这一点的优化仅针对 group 内部原始小于 64 生效
  float wei = grad_output * val;

  for (int offset=16; offset>=1; offset >>= 1) {
    wei += __shfl_down(wei, offset);
  }

  #ifdef __gfx936__
    //! 最后一步的 warp 内规约
    //! 规约区间为
    // __builtin_amdgcn_global_atomic_fadd_f32(grad_weights, grad_output * val);

    if (threadIdx.x % 32 == 0) {
      // __builtin_amdgcn_global_atomic_fadd_f32(grad_weights, wei);
      *grad_weights +=wei;
    }

    if (threadIdx.x ==0) {
      __builtin_amdgcn_global_atomic_fadd_f32(grad_sampling_location, spl.a);
      __builtin_amdgcn_global_atomic_fadd_f32(grad_sampling_location + 1, spl.b);
    }
  #else
    atomicAdd(grad_weights, grad_output * val);
    atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
    atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
  #endif
}

__device__ void bilinear_sampling_grad(
    const float *&bottom_data, const float &weight,
    const int &height, const int &width,
    const int &num_embeds, const float &h_im, const float &w_im,
    const int &base_ptr,
    const float &grad_output,
    float *&grad_mc_ms_feat, float *grad_sampling_location, float *grad_weights) {
  const int h_low = floorf(h_im);
  const int w_low = floorf(w_im);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const float lh = h_im - h_low;
  const float lw = w_im - w_low;
  const float hh = 1 - lh, hw = 1 - lw;

  const int w_stride = num_embeds;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  const float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  const float top_grad_mc_ms_feat = grad_output * weight;
  float grad_h_weight = 0, grad_w_weight = 0;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_h_weight -= hw * v1;
    grad_w_weight -= hh * v1;

    #ifdef __gfx936__
      __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
    #else
      atomicAdd(grad_mc_ms_feat + ptr1, w1 * top_grad_mc_ms_feat);
    #endif
  }

  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_h_weight -= lw * v2;
    grad_w_weight += hh * v2;

    // atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);

    #ifdef __gfx936__
      __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
    #else
      atomicAdd(grad_mc_ms_feat + ptr2, w2 * top_grad_mc_ms_feat);
    #endif

  }
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
    grad_h_weight += hw * v3;
    grad_w_weight -= lh * v3;
    // atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);

    #ifdef __gfx936__
      __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
    #else
      atomicAdd(grad_mc_ms_feat + ptr3, w3 * top_grad_mc_ms_feat);
    #endif

  }
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1) {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
    grad_h_weight += lw * v4;
    grad_w_weight += lh * v4;
    // atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);

    #ifdef __gfx936__
      __builtin_amdgcn_global_atomic_fadd_f32(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
    #else
      atomicAdd(grad_mc_ms_feat + ptr4, w4 * top_grad_mc_ms_feat);
    #endif

  }

  const float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);


  #ifdef __gfx936__
    __builtin_amdgcn_global_atomic_fadd_f32(grad_weights, grad_output * val);

    __builtin_amdgcn_global_atomic_fadd_f32(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
    __builtin_amdgcn_global_atomic_fadd_f32(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
  #else
    atomicAdd(grad_weights, grad_output * val);
    atomicAdd(grad_sampling_location, width * grad_w_weight * top_grad_mc_ms_feat);
    atomicAdd(grad_sampling_location + 1, height * grad_h_weight * top_grad_mc_ms_feat);
  #endif
}

__global__ void deformable_aggregation_kernel(
    const int64_t num_kernels,
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int64_t block_id = blockIdx.x;   // block -> (batch, anchor)
    int batch_idx = block_id / num_anchors;
    int anchor_local = block_id % num_anchors;
    int anchor_index = batch_idx * num_anchors + anchor_local;

    int channel = threadIdx.x;       // thread -> channel
    if(channel >= num_embeds) return;

    double accum = 0.0;

    for(int p=0; p<num_pts; ++p) {
        for(int cam=0; cam<num_cams; ++cam) {
            int loc_offset = (((anchor_index * num_pts + p) * num_cams + cam) << 1);
            float loc_w = sample_location[loc_offset + 0];
            float loc_h = sample_location[loc_offset + 1];
            if(!(loc_w>0.f && loc_w<1.f && loc_h>0.f && loc_h<1.f)) continue;

            for(int s=0; s<num_scale; ++s) {
                int cam_scale_index = cam * num_scale + s;
                int sp_base = cam_scale_index * 2;
                int H = spatial_shape[sp_base + 0];
                int W = spatial_shape[sp_base + 1];

                float h_im = loc_h * H - 0.5f;
                float w_im = loc_w * W - 0.5f;

                int feat_map_idx = batch_idx * num_feat + scale_start_index[cam_scale_index];
                int base_ptr = feat_map_idx * num_embeds + channel;

                float sampled = bilinear_sampling(mc_ms_feat, H, W, num_embeds, h_im, w_im, base_ptr);

                int embeds_per_group = num_embeds / num_groups;
                int group = embeds_per_group>0 ? (channel / embeds_per_group) : 0;
                int w_idx = (((((anchor_index*num_pts + p)*num_cams + cam)*num_scale + s)*num_groups)+group);
                double w_val = double(weights[w_idx]);

                accum += (double)sampled * (double)w_val;
            }
        }
    }
    float result=float(accum);

#ifdef __gfx936__
    __builtin_amdgcn_global_atomic_fadd_f32(output + anchor_index * num_embeds + channel, result);
#else
    atomicAdd(output + anchor_index * num_embeds + channel, result);
#endif
}


// 辅助函数：判断是否为 2 的幂（device 上 inline）
__device__ __forceinline__ bool is_pow2_int(int x) {
    return (x > 0) && ((x & (x - 1)) == 0);
}
// 辅助函数：计算 base-2 的对数 (假设 x 是 2 的幂)
__device__ __forceinline__ int log2_int(int x) {
    int s = 0;
    while ((x & 1) == 0) { x >>= 1; ++s; }
    return s;
}


__global__ void deformable_aggregation_grad_kernel_sp(
    const int num_kernels,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    const float* grad_output,
    float* grad_mc_ms_feat,
    float* grad_sampling_location,
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups) {
  extern __shared__ float2_t s_data[];

  int block_id = blockIdx.x;  // 每个block对应(batch, anchor)
  int batch_idx = block_id / num_anchors;
  int anchor_local = block_id % num_anchors;
  int anchor_index = batch_idx * num_anchors + anchor_local;

  int channel = threadIdx.x;  // 每个thread对应一个通道
  if (channel >= num_embeds) return;

  float grad_accum = 0.0f;

  // —— 循环计算点、相机、尺度的梯度
  for (int p = 0; p < num_pts; ++p) {
    for (int cam = 0; cam < num_cams; ++cam) {
      int loc_offset = (((anchor_index * num_pts + p) * num_cams + cam) << 1);
      float loc_w = sample_location[loc_offset + 0];
      float loc_h = sample_location[loc_offset + 1];

      if (loc_w <= 0 || loc_w >= 1 || loc_h <= 0 || loc_h >= 1)
        continue;  // 跳过越界点

      for (int s = 0; s < num_scale; ++s) {
        int cam_scale_index = cam * num_scale + s;
        int sp_base = cam_scale_index * 2;
        int H = spatial_shape[sp_base];
        int W = spatial_shape[sp_base + 1];

        float h_im = loc_h * H - 0.5f;
        float w_im = loc_w * W - 0.5f;

        int feat_map_idx = batch_idx * num_feat + scale_start_index[cam_scale_index];
        int base_ptr = feat_map_idx * num_embeds + channel;

        int embeds_per_group = num_embeds / num_groups;
        int group = embeds_per_group > 0 ? (channel / embeds_per_group) : 0;
        int w_idx = (((((anchor_index * num_pts + p) * num_cams + cam) * num_scale + s) * num_groups) + group);

        float weight_val = weights[w_idx];
        float grad_val = grad_output[anchor_index * num_embeds + channel];

        // —— 调用梯度计算
        bilinear_sampling_grad_sp(
            mc_ms_feat, weight_val, H, W, num_embeds, h_im, w_im, base_ptr,
            grad_val, grad_mc_ms_feat,
            grad_sampling_location + loc_offset,
            grad_weights + w_idx,
            s_data);
      }
    }
  }
}

__global__ void deformable_aggregation_grad_kernel(
    const int64_t num_kernels,
    const float* mc_ms_feat,       // [bs, anchor, pts, cam, scale, channel]
    const int* spatial_shape,      // [cam, scale, 2]
    const int* scale_start_index,  // [cam, scale]
    const float* sample_location,  // [bs, anchor, pts, cam, 2(y, x)]
    const float* weights,          // [bs, anchor, cam, scale, group]
    const float* grad_output,      // [bs, anchor, c]
    float* grad_mc_ms_feat,        // same as feat
    float* grad_sampling_location, // same as sampling location
    float* grad_weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_kernels) return;

    const int weights_ptr = idx / (num_embeds / num_groups);
    const int channel_index = idx % num_embeds;
    idx /= num_embeds;
    const int scale_index = idx % num_scale;
    idx /= num_scale;

    const int cam_index = idx % num_cams;
    idx /= num_cams;
    const int pts_index = idx % num_pts;
    idx /= num_pts;

    int anchor_index = idx % num_anchors;
    idx /= num_anchors;
    const int batch_index = idx % batch_size;
    idx /= batch_size;

    anchor_index = batch_index * num_anchors + anchor_index;
    const int loc_offset = ((anchor_index * num_pts + pts_index) * num_cams + cam_index) << 1;

    const float loc_w = sample_location[loc_offset];
    if (loc_w <= 0 || loc_w >= 1) return;
    const float loc_h = sample_location[loc_offset + 1];
    if (loc_h <= 0 || loc_h >= 1) return;

    const float grad = grad_output[anchor_index*num_embeds + channel_index];

    int cam_scale_index = cam_index * num_scale + scale_index;
    const int value_offset = (batch_index * num_feat + scale_start_index[cam_scale_index]) * num_embeds + channel_index;

    cam_scale_index = cam_scale_index << 1;
    const int h = spatial_shape[cam_scale_index];
    const int w = spatial_shape[cam_scale_index + 1];

    const float h_im = loc_h * h - 0.5;
    const float w_im = loc_w * w - 0.5;


    const float weight = weights[weights_ptr];
    float *grad_weights_ptr = grad_weights + weights_ptr;
    float *grad_location_ptr = grad_sampling_location + loc_offset;
    bilinear_sampling_grad(
        mc_ms_feat, weight, h, w, num_embeds, h_im, w_im,
        value_offset,
        grad,
        grad_mc_ms_feat, grad_location_ptr, grad_weights_ptr
    );
}

void deformable_aggregation(
    float* output,
    const float* mc_ms_feat,
    const int* spatial_shape,
    const int* scale_start_index,
    const float* sample_location,
    const float* weights,
    int batch_size,
    int num_cams,
    int num_feat,
    int num_embeds,
    int num_scale,
    int num_anchors,
    int num_pts,
    int num_groups
) {
    // new grid: one block per (batch, anchor)
    const int grid = batch_size * num_anchors;

    // choose block threads: try to use a power-of-two near num_embeds but <= 1024
    int threads = 256;
    if (num_embeds <= 128) threads = 128;
    else if (num_embeds < 256) threads = 256;
    else if (num_embeds <= 512) threads = 512;
    else threads = 1024;

    deformable_aggregation_kernel<<< grid, threads >>>(
      0,output, mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
        batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
    );

}

void deformable_aggregation_grad(
  const float* mc_ms_feat,
  const int* spatial_shape,
  const int* scale_start_index,
  const float* sample_location,
  const float* weights,
  const float* grad_output,
  float* grad_mc_ms_feat,
  float* grad_sampling_location,
  float* grad_weights,
  int batch_size,
  int num_cams,
  int num_feat,
  int num_embeds,
  int num_scale,
  int num_anchors,
  int num_pts,
  int num_groups
) {
    const int64_t num_kernels = (int64_t)batch_size * num_pts * num_embeds * num_anchors * num_cams * num_scale;

     if (num_embeds != 256 || ((num_embeds / num_groups) != 32)) {
          deformable_aggregation_grad_kernel
              <<<(int)ceil(((double)num_kernels/128)), 128>>>(
              num_kernels,
              mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
              grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
              batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
          );
      } else {

        int num_blocks = batch_size * num_anchors;
        int threads_per_block = num_embeds;
        deformable_aggregation_grad_kernel_sp<<<num_blocks, threads_per_block, 256 * 2 * sizeof(float)>>>(
            0,
            mc_ms_feat, spatial_shape, scale_start_index, sample_location, weights,
            grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
            batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
        );
      }
}

/* feat: bs, num_feat, c */
/* _spatial_shape: cam, scale, 2 */
/* _scale_start_index: cam, scale */
/* _sampling_location: bs, anchor, pts, cam, 2 */
/* _weights: bs, anchor, pts, cam, scale, group */
/* output: bs, anchor, c */
/* kernel: bs, anchor, pts, c */


at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();

  auto output = at::zeros({batch_size, num_anchors, num_embeds}, _mc_ms_feat.options());
  int warm_up = 10;
  int prof_cnt = 50;
  cudaEvent_t start, stop;
  float milliseconds = 0;
  deformable_aggregation(
    output.data_ptr<float>(),
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
  return output;
}

void deformable_aggregation_backward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights,
  const at::Tensor &_grad_output,
  at::Tensor &_grad_mc_ms_feat,
  at::Tensor &_grad_sampling_location,
  at::Tensor &_grad_weights
) {
  at::DeviceGuard guard(_mc_ms_feat.device());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_mc_ms_feat));
  int batch_size = _mc_ms_feat.size(0);
  int num_feat = _mc_ms_feat.size(1);
  int num_embeds = _mc_ms_feat.size(2);
  int num_cams = _spatial_shape.size(0);
  int num_scale = _spatial_shape.size(1);
  int num_anchors = _sampling_location.size(1);
  int num_pts = _sampling_location.size(2);
  int num_groups = _weights.size(5);

  const float* mc_ms_feat = _mc_ms_feat.data_ptr<float>();
  const int* spatial_shape = _spatial_shape.data_ptr<int>();
  const int* scale_start_index = _scale_start_index.data_ptr<int>();
  const float* sampling_location = _sampling_location.data_ptr<float>();
  const float* weights = _weights.data_ptr<float>();
  const float* grad_output = _grad_output.data_ptr<float>();

  float* grad_mc_ms_feat = _grad_mc_ms_feat.data_ptr<float>();
  float* grad_sampling_location = _grad_sampling_location.data_ptr<float>();
  float* grad_weights = _grad_weights.data_ptr<float>();

  auto _test_grad_mc_ms_feat_tensor = _grad_mc_ms_feat.clone();
  auto _test_grad_sampling_location_tensor = _grad_sampling_location.clone();
  auto _test_grad_weights_tensor = _grad_weights.clone();

  float* _test_grad_mc_ms_feat = _test_grad_mc_ms_feat_tensor.data_ptr<float>();
  float* _test_grad_sampling_location = _test_grad_sampling_location_tensor.data_ptr<float>();
  float* _test_grad_weights = _test_grad_weights_tensor.data_ptr<float>();

  deformable_aggregation_grad(
    mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
    grad_output, grad_mc_ms_feat, grad_sampling_location, grad_weights,
    batch_size, num_cams, num_feat, num_embeds, num_scale, num_anchors, num_pts, num_groups
  );
}
