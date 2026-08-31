// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <vector>
#include "pybind_caster.h"

namespace at {
namespace native {
Tensor grid_sample_forward(
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners);

std::tuple<Tensor,Tensor> grid_sample_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask);

Tensor upsample_bilinear_2d_forward(
    const Tensor& input,
    c10::OptionalIntArrayRef output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scales_factors);

Tensor upsample_bilinear_2d_backward(
    const Tensor& grad_output,
    c10::OptionalIntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scales_factors);

}
}

at::Tensor deformable_aggregation_forward(
  const at::Tensor &_mc_ms_feat,
  const at::Tensor &_spatial_shape,
  const at::Tensor &_scale_start_index,
  const at::Tensor &_sampling_location,
  const at::Tensor &_weights
);

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
);

void bind_bev_pool(py::module_ &m);
void bind_sparse_conv(py::module_ &m);

namespace voxelization {
void bind_voxelization(py::module_ &m);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_sample_forward", &at::native::grid_sample_forward, "grid_sample forward");
  m.def("grid_sample_backward", &at::native::grid_sample_backward, "grid_sample backward");
  m.def("upsample_bilinear_2d_forward", &at::native::upsample_bilinear_2d_forward, "upsample bilinear 2d forward",
    py::arg("input"),
    py::arg("output_size") = py::none(),
    py::arg("align_corners"),
    py::arg("scale_factors") = py::none());
  m.def("upsample_bilinear_2d_backward", &at::native::upsample_bilinear_2d_backward, "upsample bilinear 2d backward");
  m.def("deformable_aggregation_forward", &deformable_aggregation_forward, "deformable_aggregation_forward");
  m.def("deformable_aggregation_backward", &deformable_aggregation_backward, "deformable_aggregation_backward");
  bind_bev_pool(m);
  voxelization::bind_voxelization(m);
  bind_sparse_conv(m);
}
