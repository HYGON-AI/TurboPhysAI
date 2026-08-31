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

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

// CUDA function declarations
void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad);

void bev_pool_prepare(int total, int b, int geom_d, int geom_h, int geom_w,
  int out_d, int out_h, int out_w, const float* geom_feats, const float* bx,
  const float* dx, const int64_t* nx, int* out_coords, int* out_ranks,
  bool* out_kept);

void bev_pool_prepare_geometry(int total, int b, int ncam, int geom_d,
  int geom_h, int geom_w, int out_d, int out_h, int out_w,
  const float* frustum, const float* inv_post_rots, const float* post_trans,
  const float* combine, const float* camera2lidar_trans,
  const float* extra_rots, const float* extra_trans, const float* bx,
  const float* dx, const int64_t* nx, float boundary_eps, int* out_coords,
  int* out_ranks, bool* out_kept, bool* out_boundary);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    out              : output features, FloatTensor[b, d, h, w, c]
*/
at::Tensor bev_pool_forward(
  const at::Tensor _x,
  const at::Tensor _geom_feats, 
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _x.size(0);
  int c = _x.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_x));
  const float* x = _x.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();
  
  auto options =
      torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  at::Tensor _out = torch::zeros({b, d, h, w, c}, options);
  float* out = _out.data_ptr<float>();
  bev_pool(
    b, d, h, w, n, c, n_intervals, x,
    geom_feats, interval_starts, interval_lengths, out
  );
  return _out;
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : input features, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    x_grad           : output features, FloatTensor[n, 4]
*/
at::Tensor bev_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_feats, 
  const at::Tensor _interval_lengths, 
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_feats.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _x_grad = torch::zeros({n, c}, options);
  float* x_grad = _x_grad.data_ptr<float>();
  
  bev_pool_grad(
    b, d, h, w, n, c, n_intervals, out_grad,
    geom_feats, interval_starts, interval_lengths, x_grad
  );
  
  return _x_grad;
}

std::vector<at::Tensor> bev_pool_prepare_forward(
  const at::Tensor _geom_feats,
  const at::Tensor _bx,
  const at::Tensor _dx,
  const at::Tensor _nx,
  int b, int d, int h, int w
) {
  TORCH_CHECK(_geom_feats.is_cuda(), "geom_feats must be a CUDA/HIP tensor");
  TORCH_CHECK(_geom_feats.scalar_type() == at::ScalarType::Float,
              "geom_feats must be float32");
  TORCH_CHECK(_geom_feats.dim() == 6 && _geom_feats.size(5) == 3,
              "geom_feats must have shape [B, N, D, H, W, 3]");
  TORCH_CHECK(_bx.numel() == 3 && _dx.numel() == 3 && _nx.numel() == 3,
              "bx, dx and nx must have 3 elements");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_geom_feats));
  int total = _geom_feats.numel() / 3;
  int geom_d = _geom_feats.size(2);
  int geom_h = _geom_feats.size(3);
  int geom_w = _geom_feats.size(4);

  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(_geom_feats.device());
  auto bool_options =
      torch::TensorOptions().dtype(torch::kBool).device(_geom_feats.device());
  at::Tensor coords = torch::empty({total, 4}, int_options);
  at::Tensor ranks = torch::empty({total}, int_options);
  at::Tensor kept = torch::empty({total}, bool_options);

  bev_pool_prepare(
    total, b, geom_d, geom_h, geom_w, d, h, w,
    _geom_feats.data_ptr<float>(), _bx.data_ptr<float>(), _dx.data_ptr<float>(),
    _nx.data_ptr<int64_t>(), coords.data_ptr<int>(), ranks.data_ptr<int>(),
    kept.data_ptr<bool>()
  );
  return {coords, ranks, kept};
}

std::vector<at::Tensor> bev_pool_prepare_geometry_forward(
  const at::Tensor _frustum,
  const at::Tensor _inv_post_rots,
  const at::Tensor _post_trans,
  const at::Tensor _combine,
  const at::Tensor _camera2lidar_trans,
  const at::Tensor _extra_rots,
  const at::Tensor _extra_trans,
  const at::Tensor _bx,
  const at::Tensor _dx,
  const at::Tensor _nx,
  int b, int d, int h, int w,
  double boundary_eps
) {
  TORCH_CHECK(_frustum.is_cuda(), "frustum must be a CUDA/HIP tensor");
  TORCH_CHECK(_frustum.scalar_type() == at::ScalarType::Float,
              "frustum must be float32");
  TORCH_CHECK(_frustum.dim() == 4 && _frustum.size(3) == 3,
              "frustum must have shape [D, H, W, 3]");
  TORCH_CHECK(_inv_post_rots.dim() == 4 && _inv_post_rots.size(2) == 3 &&
              _inv_post_rots.size(3) == 3, "inv_post_rots must be [B, N, 3, 3]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(_frustum));
  int ncam = _inv_post_rots.size(1);
  int geom_d = _frustum.size(0);
  int geom_h = _frustum.size(1);
  int geom_w = _frustum.size(2);
  int total = b * ncam * geom_d * geom_h * geom_w;

  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(_frustum.device());
  auto bool_options =
      torch::TensorOptions().dtype(torch::kBool).device(_frustum.device());
  at::Tensor coords = torch::empty({total, 4}, int_options);
  at::Tensor ranks = torch::empty({total}, int_options);
  at::Tensor kept = torch::empty({total}, bool_options);
  at::Tensor boundary = torch::empty({total}, bool_options);

  bev_pool_prepare_geometry(
    total, b, ncam, geom_d, geom_h, geom_w, d, h, w,
    _frustum.data_ptr<float>(), _inv_post_rots.data_ptr<float>(),
    _post_trans.data_ptr<float>(), _combine.data_ptr<float>(),
    _camera2lidar_trans.data_ptr<float>(), _extra_rots.data_ptr<float>(),
    _extra_trans.data_ptr<float>(), _bx.data_ptr<float>(), _dx.data_ptr<float>(),
    _nx.data_ptr<int64_t>(), static_cast<float>(boundary_eps), coords.data_ptr<int>(),
    ranks.data_ptr<int>(), kept.data_ptr<bool>(), boundary.data_ptr<bool>()
  );
  return {coords, ranks, kept, boundary};
}

void bind_bev_pool(py::module_ &m) {
  m.def("bev_pool_forward", &bev_pool_forward,
        "bev_pool_forward");
  m.def("bev_pool_backward", &bev_pool_backward,
        "bev_pool_backward");
  m.def("bev_pool_prepare", &bev_pool_prepare_forward,
        "bev_pool_prepare");
  m.def("bev_pool_prepare_geometry", &bev_pool_prepare_geometry_forward,
        "bev_pool_prepare_geometry");
}
