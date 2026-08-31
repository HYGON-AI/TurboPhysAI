// Copyright 2018-2019 OpenMMLab. All rights reserved.
// Copyright 2026 Hygon Information Technology Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modified by Hygon.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

/*
  Function: pillar pooling
  Args:
    b                : batch size
    d                : depth of the feature map
    h                : height of pooled feature map
    w                : width of pooled feature map
    n                : number of input points
    c                : number of channels
    n_intervals      : number of unique points
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
__global__ void bev_pool_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                  const float *__restrict__ x,
                                  const int *__restrict__ geom_feats,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_geom_feats = geom_feats + interval_start * 4;
  const float* cur_x = x + interval_start * c + cur_c;
  float* cur_out = out + cur_geom_feats[3] * d * h * w * c + 
    cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c + 
    cur_geom_feats[1] * c + cur_c;
  float psum = 0;
  for(int i = 0; i < interval_length; i++){
    psum += cur_x[i * c];
  }
  *cur_out = psum;
}


/*
  Function: pillar pooling backward
  Args:
    b                : batch size
    d                : depth of the feature map
    h                : height of pooled feature map
    w                : width of pooled feature map
    n                : number of input points
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    x_grad           : gradient of the image fmap, FloatTensor
*/
__global__ void bev_pool_grad_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                  const float *__restrict__ out_grad,
                                  const int *__restrict__ geom_feats,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ x_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  
  const int* cur_geom_feats = geom_feats + interval_start * 4;
  float* cur_x_grad = x_grad + interval_start * c + cur_c;
  
  const float* cur_out_grad = out_grad + cur_geom_feats[3] * d * h * w * c + 
    cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c + 
    cur_geom_feats[1] * c + cur_c;
  for(int i = 0; i < interval_length; i++){
    cur_x_grad[i * c] = *cur_out_grad;
  }
  
}

__global__ void __launch_bounds__(256) bev_pool_prepare_kernel(
    int total, int b, int geom_d, int geom_h, int geom_w, int out_d, int out_h,
    int out_w, const float* __restrict__ geom_feats,
    const float* __restrict__ bx, const float* __restrict__ dx,
    const int64_t* __restrict__ nx, int* __restrict__ out_coords,
    int* __restrict__ out_ranks, bool* __restrict__ out_kept) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int t = idx;
  int w_idx = t % geom_w;
  t /= geom_w;
  int h_idx = t % geom_h;
  t /= geom_h;
  int d_idx = t % geom_d;
  t /= geom_d;
  int n_idx = t % (total / (b * geom_d * geom_h * geom_w));
  t /= (total / (b * geom_d * geom_h * geom_w));
  int b_idx = t;
  (void)d_idx;
  (void)h_idx;
  (void)w_idx;

  const float* cur_geom = geom_feats + idx * 3;
  float half = 0.5f;
  int cx = (int)((cur_geom[0] - (bx[0] - dx[0] * half)) / dx[0]);
  int cy = (int)((cur_geom[1] - (bx[1] - dx[1] * half)) / dx[1]);
  int cz = (int)((cur_geom[2] - (bx[2] - dx[2] * half)) / dx[2]);
  bool kept = (cx >= 0) && (cx < nx[0]) &&
              (cy >= 0) && (cy < nx[1]) &&
              (cz >= 0) && (cz < nx[2]);

  int* cur_out = out_coords + idx * 4;
  cur_out[0] = cx;
  cur_out[1] = cy;
  cur_out[2] = cz;
  cur_out[3] = b_idx;
  out_ranks[idx] = kept ? (cx * (out_w * out_d * b) + cy * (out_d * b) +
                           cz * b + b_idx) : -1;
  out_kept[idx] = kept;
}

__device__ __forceinline__ bool bev_pool_near_int_boundary(float value,
                                                           float eps) {
  float lower = floorf(value);
  float upper = lower + 1.0f;
  return fabsf(value - lower) <= eps || fabsf(upper - value) <= eps;
}

__global__ void __launch_bounds__(256) bev_pool_prepare_geometry_kernel(
    int total, int b, int ncam, int geom_d, int geom_h, int geom_w, int out_d,
    int out_h, int out_w, const float* __restrict__ frustum,
    const float* __restrict__ inv_post_rots,
    const float* __restrict__ post_trans, const float* __restrict__ combine,
    const float* __restrict__ camera2lidar_trans,
    const float* __restrict__ extra_rots, const float* __restrict__ extra_trans,
    const float* __restrict__ bx, const float* __restrict__ dx,
    const int64_t* __restrict__ nx, float boundary_eps,
    int* __restrict__ out_coords, int* __restrict__ out_ranks,
    bool* __restrict__ out_kept, bool* __restrict__ out_boundary) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  int t = idx;
  int w_idx = t % geom_w;
  t /= geom_w;
  int h_idx = t % geom_h;
  t /= geom_h;
  int d_idx = t % geom_d;
  t /= geom_d;
  int n_idx = t % ncam;
  int b_idx = t / ncam;

  const float* fr = frustum + ((d_idx * geom_h + h_idx) * geom_w + w_idx) * 3;
  const int bn = b_idx * ncam + n_idx;
  const float* inv_post = inv_post_rots + bn * 9;
  const float* post_t = post_trans + bn * 3;
  const float* comb = combine + bn * 9;
  const float* cam_t = camera2lidar_trans + bn * 3;
  const float* er = extra_rots + b_idx * 9;
  const float* et = extra_trans + b_idx * 3;

  float px = fr[0] - post_t[0];
  float py = fr[1] - post_t[1];
  float pz = fr[2] - post_t[2];

  float qx = inv_post[0] * px + inv_post[1] * py + inv_post[2] * pz;
  float qy = inv_post[3] * px + inv_post[4] * py + inv_post[5] * pz;
  float qz = inv_post[6] * px + inv_post[7] * py + inv_post[8] * pz;

  float rx = qx * qz;
  float ry = qy * qz;
  float rz = qz;

  float sx = comb[0] * rx + comb[1] * ry + comb[2] * rz + cam_t[0];
  float sy = comb[3] * rx + comb[4] * ry + comb[5] * rz + cam_t[1];
  float sz = comb[6] * rx + comb[7] * ry + comb[8] * rz + cam_t[2];

  float gx = er[0] * sx + er[1] * sy + er[2] * sz + et[0];
  float gy = er[3] * sx + er[4] * sy + er[5] * sz + et[1];
  float gz = er[6] * sx + er[7] * sy + er[8] * sz + et[2];

  const float half = 0.5f;
  float ux = (gx - (bx[0] - dx[0] * half)) / dx[0];
  float uy = (gy - (bx[1] - dx[1] * half)) / dx[1];
  float uz = (gz - (bx[2] - dx[2] * half)) / dx[2];
  int cx = (int)ux;
  int cy = (int)uy;
  int cz = (int)uz;
  bool kept = (cx >= 0) && (cx < nx[0]) &&
              (cy >= 0) && (cy < nx[1]) &&
              (cz >= 0) && (cz < nx[2]);

  int* cur_out = out_coords + idx * 4;
  cur_out[0] = cx;
  cur_out[1] = cy;
  cur_out[2] = cz;
  cur_out[3] = b_idx;
  out_ranks[idx] = kept ? (cx * (out_w * out_d * b) + cy * (out_d * b) +
                           cz * b + b_idx) : -1;
  out_kept[idx] = kept;
  out_boundary[idx] = bev_pool_near_int_boundary(ux, boundary_eps) ||
                      bev_pool_near_int_boundary(uy, boundary_eps) ||
                      bev_pool_near_int_boundary(uz, boundary_eps);
}

void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out) {
  bev_pool_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts, interval_lengths, out
  );
}

void bev_pool_prepare(int total, int b, int geom_d, int geom_h, int geom_w,
  int out_d, int out_h, int out_w, const float* geom_feats, const float* bx,
  const float* dx, const int64_t* nx, int* out_coords, int* out_ranks,
  bool* out_kept) {
  bev_pool_prepare_kernel<<<(int)ceil(((double)total / 256)), 256>>>(
    total, b, geom_d, geom_h, geom_w, out_d, out_h, out_w, geom_feats, bx, dx,
    nx, out_coords, out_ranks, out_kept
  );
}

void bev_pool_prepare_geometry(int total, int b, int ncam, int geom_d,
  int geom_h, int geom_w, int out_d, int out_h, int out_w,
  const float* frustum, const float* inv_post_rots, const float* post_trans,
  const float* combine, const float* camera2lidar_trans,
  const float* extra_rots, const float* extra_trans, const float* bx,
  const float* dx, const int64_t* nx, float boundary_eps, int* out_coords,
  int* out_ranks, bool* out_kept, bool* out_boundary) {
  bev_pool_prepare_geometry_kernel<<<(int)ceil(((double)total / 256)), 256>>>(
    total, b, ncam, geom_d, geom_h, geom_w, out_d, out_h, out_w, frustum,
    inv_post_rots, post_trans, combine, camera2lidar_trans, extra_rots,
    extra_trans, bx, dx, nx, boundary_eps, out_coords, out_ranks, out_kept, out_boundary
  );
}

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad) {
  bev_pool_grad_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    b, d, h, w, n, c, n_intervals, out_grad, geom_feats, interval_starts, interval_lengths, x_grad
  );
}
