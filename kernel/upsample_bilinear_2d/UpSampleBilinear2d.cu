// Derived from PyTorch: aten/src/ATen/native/cuda/UpSampleBilinear2d.cu
// PyTorch v2.7.1, commit e2d141dbde55c2a4370fac5165b0561b6af4798b.
// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon.
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// #include <ATen/core/Tensor.h>
// #include <ATen/AccumulateType.h>
// #include <ATen/ceil_div.h>
// #include <ATen/Dispatch.h>
// #include <ATen/TensorUtils.h>
// #include <ATen/Utils.h>
// #include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/UpSample.cuh>
// #include <ATen/native/cuda/KernelUtils.cuh>
// #include <ATen/cuda/detail/KernelUtils.h>
// #include <ATen/native/cuda/LaunchUtils.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "hip/hip_fp16.h"
#include "hip/hip_bf16.h"
#include "hip/hip_runtime.h"


// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// #include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
// #include <ATen/ops/_upsample_bicubic2d_aa_native.h>
// #include <ATen/ops/_upsample_bilinear2d_aa_backward_native.h>
// #include <ATen/ops/_upsample_bilinear2d_aa_native.h>
// #include <ATen/ops/empty.h>
// #include <ATen/ops/upsample_bilinear2d_backward_native.h>
// #include <ATen/ops/upsample_bilinear2d_native.h>
// #include <ATen/ops/zeros.h>
// #endif

#include "utils.h"

// #if defined(USE_ROCM) && ROCM_VERSION > 60100
// #define GRID_TYPE uint32_t*
// #else
// #define GRID_TYPE int*
// #endif

using at::native::upsample::compute_output_size;

namespace at {
namespace native {

// template <typename scalar_t, typename accscalar_t>
// C10_LAUNCH_BOUNDS_1(1024)
// __global__ void upsample_bilinear2d_out_frame(
//     const int n,
//     const accscalar_t rheight,
//     const accscalar_t rwidth,
//     const bool align_corners,
//     const PackedTensorAccessor<scalar_t, 4> idata,
//     PackedTensorAccessor<scalar_t, 4> odata) {
//   int index = threadIdx.x + blockIdx.x * blockDim.x;

//   const int batchsize = idata.size(0);
//   const int channels = idata.size(1);
//   const int height1 = idata.size(2);
//   const int width1 = idata.size(3);
//   const int width2 = odata.size(3);

//   if (index < n) {
//     const int w2 = index % width2; // 0:width2-1
//     const int h2 = index / width2; // 0:height2-1

//     const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
//         rheight, h2, align_corners, /*cubic=*/false);
//     const int h1 = h1r;
//     const int h1p = (h1 < height1 - 1) ? 1 : 0;
//     const accscalar_t h1lambda = h1r - h1;
//     const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
//     //
//     const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
//         rwidth, w2, align_corners, /*cubic=*/false);
//     const int w1 = w1r;
//     const int w1p = (w1 < width1 - 1) ? 1 : 0;
//     const accscalar_t w1lambda = w1r - w1;
//     const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
//     //
//     for (int n = 0; n < batchsize; n++) {
//       for (int c = 0; c < channels; ++c) {
//         const accscalar_t val = h0lambda *
//                 (w0lambda * idata[n][c][h1][w1] +
//                  w1lambda * idata[n][c][h1][w1 + w1p]) +
//             h1lambda *
//                 (w0lambda * idata[n][c][h1 + h1p][w1] +
//                  w1lambda * idata[n][c][h1 + h1p][w1 + w1p]);
//         odata[n][c][h2][w2] = static_cast<scalar_t>(val);
//       }
//     }
//   }
// }

// template <typename scalar_t, typename accscalar_t>
// C10_LAUNCH_BOUNDS_1(1024)
// __global__ void upsample_bilinear2d_nhwc_out_frame(
//     const accscalar_t rheight,
//     const accscalar_t rwidth,
//     const bool align_corners,
//     const int channels,
//     const int height1,
//     const int width1,
//     const int height2,
//     const int width2,
//     const scalar_t* idata,
//     scalar_t* odata,
//     const int out_numel) {

//   const int index = blockIdx.x * blockDim.x + threadIdx.x;

//   if (index < out_numel) {
//     const int c = index % channels;
//     const int w2 = (index / channels) % width2;
//     const int h2 = (index / channels / width2) % height2;
//     const int n = index / channels / width2 / height2;

//     const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
//         rheight, h2, align_corners, /*cubic=*/false);
//     const int h1 = h1r;
//     const int h1p = (h1 < height1 - 1) ? 1 : 0;
//     const accscalar_t h1lambda = h1r - h1;
//     const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

//     const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
//         rwidth, w2, align_corners, /*cubic=*/false);
//     const int w1 = w1r;
//     const int w1p = (w1 < width1 - 1) ? 1 : 0;
//     const accscalar_t w1lambda = w1r - w1;
//     const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

//     const accscalar_t val = h0lambda * (
//         w0lambda * idata[idx_cl(n, h1, w1, c, height1, width1, channels)] +
//         w1lambda * idata[idx_cl(n, h1, w1 + w1p, c, height1, width1, channels)]
//       ) + h1lambda * (
//         w0lambda * idata[idx_cl(n, h1 + h1p, w1, c, height1, width1, channels)] +
//         w1lambda * idata[idx_cl(n, h1 + h1p, w1 + w1p, c, height1, width1, channels)]
//       );
//     odata[idx_cl(n, h2, w2, c, height2, width2, channels)] = static_cast<scalar_t>(val);
//   }
// }

// // Backward (adjoint) operation 1 <- 2 (accumulates)
// template <typename scalar_t, typename accscalar_t>
// __global__ void upsample_bilinear2d_backward_out_frame(
//     const size_t nc,
//     const int height1,
//     const int width1,
//     const int height2,
//     const int width2,
//     const accscalar_t rheight,
//     const accscalar_t rwidth,
//     const bool align_corners,
//     scalar_t* __restrict__ idata,
//     const scalar_t* __restrict__ odata) {
//   const size_t o_numel = nc * width2 * height2;
//   const size_t i_numel = nc * width1 * height1;
//   size_t index = blockDim.x * blockIdx.x + threadIdx.x;
//   size_t index_temp = index;
//   const int output_block_offset = (index/(height1*width1)) * height2 * width2;
//   const int ih = (index_temp / width1)%height1;
//   const int iw = index_temp % width1;

//   //const int ihw = (blockIdx.x * height1 + ih)*width1 + iw;
//   const size_t ihw = (blockIdx.x * height1 + ih)*width1 + iw;
//   if (ih >= height1 || iw >= width1) return;

//   float total_grad = 0.0f;

//   // 缩放因子
//   const float scale_h = rheight;//output_height / (float)input_height;
//   const float scale_w = rwidth;//output_width / (float)input_width;

//   // 修正：搜索范围是3×3区域
//   int oh_start, oh_end, ow_start, ow_end;

//   if (align_corners) {
//     // 对于align_corners=true
//     oh_start = max(0, (int)ceilf((ih-1) / scale_h));
//     oh_end = min(height2-1, (int)ceilf((ih + 2) / scale_h+3));
//     ow_start = max(0, (int)ceilf((iw-1) / scale_w));
//     ow_end = min(width2-1, (int)ceilf((iw + 2) / scale_w+3));
//     for (int oh = oh_start; oh <= oh_end; oh++) {
//       for (int ow = ow_start; ow <= ow_end; ow++) {
//         // 坐标映射
//         float x, y;
//         x = oh * scale_h;
//         y = ow * scale_w;
//         //if(x<0)
//         x=(x < 0) ? 0.0:x;
//         //if(y<0)
//         y=(y < 0) ? 0.0:y;
//         //}

//         const int x0 = x;//(int)floorf(x);
//         const int y0 = y;//(int)floorf(y);
        
//         const int x1p = (x0 < height1 - 1) ? 1 : 0;
//         const int y1p = (y0 < width1 -1) ? 1 : 0;

//         // 检查当前输入点是否是四个插值点之一
//         // 注意：这里检查的是精确匹配，不是范围检查
//         const float dx = x - x0;
//         const float dy = y - y0;
//         if(idx(blockIdx.x,height1,width1,x0,y0) == ihw){
//             total_grad += odata[output_block_offset+oh * width2 + ow] * (1-dx) * (1-dy);
//         }
//         if(idx(blockIdx.x,height1,width1,x0,y0+y1p) == ihw){
//             total_grad += odata[output_block_offset+oh * width2 + ow] * (1-dx) * dy;
//         }
//         if(idx(blockIdx.x,height1,width1,x0+x1p,y0) == ihw){
//             total_grad += odata[output_block_offset+oh * width2 + ow] * dx * (1-dy);
//         }
//         if(idx(blockIdx.x,height1,width1,x0+x1p,y0+y1p) == ihw){
//             total_grad += odata[output_block_offset+oh * width2 + ow] * dx * dy;
//         }
//       }
//     }
//     idata[index] = total_grad;
//   } else {
//     // 对于align_corners=false
//     oh_start = max(0, (int)ceilf((ih-0.5) / scale_h - 0.5));
//     oh_end = min(height2-1, (int)ceilf((ih + 1.5) / scale_h-1.5));
//     ow_start = max(0, (int)ceilf((iw-0.5) / scale_w - 0.5));
//     ow_end = min(width2-1, (int)ceilf((iw + 1.5) / scale_w-1.5));
//     for (int oh = oh_start; oh <= oh_end; oh++) {
//       for (int ow = ow_start; ow <= ow_end; ow++) {
//         // 坐标映射
//         float x, y;
//         x = (oh + 0.5f) *scale_h - 0.5f;
//         y = (ow + 0.5f) *scale_w - 0.5f;
//             //if(x<0)
//         x=(x < 0) ? 0.0:x;
//             //if(y<0)
//         y=(y < 0) ? 0.0:y;
//         //}

//         const int x0 = x;//(int)floorf(x);
//         const int y0 = y;//(int)floorf(y);
        
//         const int x1p = (x0 < height1 - 1) ? 1 : 0;
//         const int y1p = (y0 < width1 -1) ? 1 : 0;

//         // 检查当前输入点是否是四个插值点之一
//         // 注意：这里检查的是精确匹配，不是范围检查
//         const float dx = x - x0;
//         const float dy = y - y0;
//         if(idx(blockIdx.x,height1,width1,x0,y0) == ihw){
//           if (index == 48754367 || index == 2000) {
//             printf("ihw = %zu (0x%zx)\n", ihw, ihw);
//             printf("idx1_val = %zu (0x%zx)\n", idx(blockIdx.x,height1,width1,x0,y0), idx(blockIdx.x,height1,width1,x0,y0));
//           }
//           total_grad += odata[output_block_offset+oh * width2 + ow] * (1-dx) * (1-dy);
//         }
//         if(idx(blockIdx.x,height1,width1,x0,y0+y1p) == ihw){
//           if (index == 48754367 || index == 2000) {
//             printf("ihw = %zu (0x%zx)\n", ihw, ihw);
//             printf("idx2_val = %zu (0x%zx)\n", idx(blockIdx.x,height1,width1,x0,y0+y1p), idx(blockIdx.x,height1,width1,x0,y0+y1p));
//           }
//           total_grad += odata[output_block_offset+oh * width2 + ow] * (1-dx) * dy;
//         }
//         if(idx(blockIdx.x,height1,width1,x0+x1p,y0) == ihw){
//           if (index == 48754367 || index == 2000) {
//             printf("ihw = %zu (0x%zx)\n", ihw, ihw);
//             printf("idx3_val = %zu (0x%zx)\n", idx(blockIdx.x,height1,width1,x0+x1p,y0), idx(blockIdx.x,height1,width1,x0+x1p,y0));
//           }
//           total_grad += odata[output_block_offset+oh * width2 + ow] * dx * (1-dy);
//         }
//         if(idx(blockIdx.x,height1,width1,x0+x1p,y0+y1p) == ihw){
//           if (index == 48754367 || index == 2000) {
//             printf("ihw = %zu (0x%zx)\n", ihw, ihw);
//             printf("idx4_val = %zu (0x%zx)\n", idx(blockIdx.x,height1,width1,x0+x1p,y0+y1p), idx(blockIdx.x,height1,width1,x0+x1p,y0+y1p));
//           }
//           total_grad += odata[output_block_offset+oh * width2 + ow] * dx * dy;
//         }
//         if (index == 48754367) {
//           printf("index: %d, ih: %d, iw: %d, oh: %d, ow: %d, x: %f, y: %f, dx: %f, dy: %f\n", index, ih, iw, oh, ow, x, y, dx, dy);
//           printf("total_grad: %f\n", total_grad);
//           printf("ihw: %zu (0x%zx), idx1: %zu (0x%zx), idx2: %zu (0x%zx), idx3: %zu (0x%zx), idx4: %zu (0x%zx)\n", ihw, ihw,
//             idx(blockIdx.x,height1,width1,x0,y0),idx(blockIdx.x,height1,width1,x0,y0),
//             idx(blockIdx.x,height1,width1,x0,y0+y1p),idx(blockIdx.x,height1,width1,x0,y0+y1p),
//             idx(blockIdx.x,height1,width1,x0+x1p,y0),idx(blockIdx.x,height1,width1,x0+x1p,y0),
//             idx(blockIdx.x,height1,width1,x0+x1p,y0+y1p),idx(blockIdx.x,height1,width1,x0+x1p,y0+y1p));
//           printf("odata values: %f, %f, %f, %f\n", odata[output_block_offset+oh * width2 + ow] * (1-dx) * (1-dy), odata[output_block_offset+oh * width2 + ow] * (1-dx) * dy, odata[output_block_offset+oh * width2 + ow] * dx * (1-dy), odata[output_block_offset+oh * width2 + ow] * dx * dy);
//         }
//       }
//     }
//     if (index == 48754367) {
//       printf("index: %d, total_grad: %f\n", index, total_grad);
//     }
//     idata[index] = total_grad;
//   }
// }

// template <typename scalar_t, typename accscalar_t>
// C10_LAUNCH_BOUNDS_1(1024)
// __global__ void upsample_bilinear2d_backward_nhwc_out_frame(
//     const int height1,
//     const int width1,
//     const int height2,
//     const int width2,
//     const accscalar_t rheight,
//     const accscalar_t rwidth,
//     const bool align_corners,
//     scalar_t* __restrict__ idata,
//     const scalar_t* __restrict__ odata,
//     const int channels,
//     const size_t o_numel,
//     const size_t i_numel) {

//   const int index = blockIdx.x * blockDim.x + threadIdx.x;

//   if (index < o_numel) {
//     const int c = index % channels;
//     const int w2 = (index / channels) % width2;
//     const int h2 = (index / channels / width2) % height2;
//     const int n = index / channels / width2 / height2;

//     const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
//         rheight, h2, align_corners, /*cubic=*/false);
//     const int h1 = h1r;
//     const int h1p = (h1 < height1 - 1) ? 1 : 0;
//     const accscalar_t h1lambda = h1r - h1;
//     const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;

//     const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
//         rwidth, w2, align_corners, /*cubic=*/false);
//     const int w1 = w1r;
//     const int w1p = (w1 < width1 - 1) ? 1 : 0;
//     const accscalar_t w1lambda = w1r - w1;
//     const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;

//     const scalar_t d2val = odata[index];
//     fastAtomicAdd(
//         idata,
//         idx_cl(n, h1, w1, c, height1, width1, channels),
//         i_numel,
//         static_cast<scalar_t>(h0lambda * w0lambda * d2val),
//         true);
//     fastAtomicAdd(
//         idata,
//         idx_cl(n, h1, w1 + w1p, c, height1, width1, channels),
//         i_numel,
//         static_cast<scalar_t>(h0lambda * w1lambda * d2val),
//         true);
//     fastAtomicAdd(
//         idata,
//         idx_cl(n, h1 + h1p, w1, c, height1, width1, channels),
//         i_numel,
//         static_cast<scalar_t>(h1lambda * w0lambda * d2val),
//         true);
//     fastAtomicAdd(
//         idata,
//         idx_cl(n, h1 + h1p, w1 + w1p, c, height1, width1, channels),
//         i_numel,
//         static_cast<scalar_t>(h1lambda * w1lambda * d2val),
//         true);
//   }
// }

// static void upsample_bilinear2d_out_cuda_template(
//     const Tensor& output,
//     const Tensor& input,
//     IntArrayRef output_size,
//     bool align_corners,
//     c10::optional<double> scales_h,
//     c10::optional<double> scales_w) {
//   TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2};
//   checkAllSameGPU(__func__, {input_arg, output_arg});

//   int output_height = output_size[0];
//   int output_width = output_size[1];

//   int channels = input.size(1);
//   int input_height = input.size(2);
//   int input_width = input.size(3);

//   const auto memory_format = input.suggest_memory_format();

//   if (input.sizes() == output.sizes()) {
//     output.copy_(input);
//     return;
//   }

//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
//     // heuristic: only use channels_last path when it's faster than the contiguous path
//     if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 16 && \
//           output.is_contiguous(memory_format)) {
//       using accscalar_t = at::acc_type<scalar_t, true>;

//       TORCH_CHECK(input.numel() < std::numeric_limits<int>::max(),
//         "upsample_bilinear2d_nhwc only supports input tensors with less than INT_MAX elements");
//       TORCH_CHECK(output.numel() < std::numeric_limits<int>::max(),
//         "upsample_bilinear2d_nhwc only supports output tensors with less than INT_MAX elements");

//       const int channels = input.size(1);
//       const int height1 = input.size(2);
//       const int width1 = input.size(3);
//       const int height2 = output.size(2);
//       const int width2 = output.size(3);

//       // const int num_kernels = output_height * output_width;
//       const int num_kernels = output.numel();
//       const int num_threads = std::min(
//           at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);

//       at::Tensor input_cl = input.contiguous(at::MemoryFormat::ChannelsLast);

//       const scalar_t* idata = input_cl.data_ptr<scalar_t>();
//       scalar_t* odata = output.data_ptr<scalar_t>();

//       const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
//           input_height, output_height, align_corners, scales_h);
//       const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
//           input_width, output_width, align_corners, scales_w);

//       upsample_bilinear2d_nhwc_out_frame<scalar_t, accscalar_t>
//         <<<ceil_div(num_kernels, num_threads), num_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
//           rheight, rwidth, align_corners,
//           channels,
//           height1,
//           width1,
//           height2,
//           width2,
//           idata, odata,
//           output.numel());
//       C10_CUDA_KERNEL_LAUNCH_CHECK();
//     } else {
//       // non-channels_last case, not necessarily contiguous
//       const int num_kernels = output_height * output_width;
//       const int num_threads = std::min(
//           at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
//       cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//       using accscalar_t = at::acc_type<scalar_t, true>;

//       auto idata = input.packed_accessor64<scalar_t, 4>();
//       auto odata = output.packed_accessor64<scalar_t, 4>();

//       const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
//           input_height, output_height, align_corners, scales_h);
//       const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
//           input_width, output_width, align_corners, scales_w);

//       upsample_bilinear2d_out_frame<scalar_t, accscalar_t>
//           <<<ceil_div(num_kernels, num_threads),
//              num_threads,
//              0,
//              stream>>>(
//               num_kernels, rheight, rwidth, align_corners, idata, odata);
//       C10_CUDA_KERNEL_LAUNCH_CHECK();
//     }
//   });
// }

// static void upsample_bilinear2d_backward_out_cuda_template(
//     const Tensor& grad_input,
//     const Tensor& grad_output_,
//     IntArrayRef output_size,
//     IntArrayRef input_size,
//     bool align_corners,
//     c10::optional<double> scales_h,
//     c10::optional<double> scales_w) {
//   TensorArg grad_input_arg{grad_input, "grad_input", 1},
//       grad_output_arg{grad_output_, "grad_output_", 2};
//   checkAllSameGPU(__func__, {grad_output_arg, grad_input_arg});

//   int output_height = output_size[0];
//   int output_width = output_size[1];

//   int nbatch = input_size[0];
//   int channels = input_size[1];
//   int input_height = input_size[2];
//   int input_width = input_size[3];

//   if (grad_input.numel() == 0) {
//     return;
//   }

//   const auto memory_format = grad_output_.suggest_memory_format();

//   // initialization to zero is required here. As we launch one thread per output
//   // element, and atomicAdd to input gradient. Given a sparse sampling case, our
//   // threads are not covering the whole input tensor.
//   grad_input.zero_();

//   // const size_t num_kernels = nbatch * channels * output_height * output_width;
//   // const int num_threads = std::min(
//   //     at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
//   cudaStream_t stream = at::cuda::getCurrentCUDAStream();

//   if (grad_output_.sizes() == grad_input.sizes()) {
//     grad_input.copy_(grad_output_);
//     return;
//   }

//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output_.scalar_type(), "upsample_bilinear2d_backward_out_frame", [&] {
//     if (memory_format == at::MemoryFormat::ChannelsLast && channels >= 4 && \
//           grad_input.is_contiguous(memory_format)) {
//       using accscalar_t = at::acc_type<scalar_t, true>;
//       const size_t num_kernels = nbatch * channels * output_height * output_width;
//       const int num_threads = std::min(
//       at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock, 1024);
//       Tensor grad_output = grad_output_.contiguous(at::MemoryFormat::ChannelsLast);

//       auto idata = grad_input.data_ptr<scalar_t>();
//       auto odata = grad_output.data_ptr<scalar_t>();

//       const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
//           input_height, output_height, align_corners, scales_h);
//       const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
//           input_width, output_width, align_corners, scales_w);

//       upsample_bilinear2d_backward_nhwc_out_frame<scalar_t, accscalar_t>
//           <<<ceil_div(num_kernels, static_cast<size_t>(num_threads)), num_threads, 0, stream>>>(
//               input_height,
//               input_width,
//               output_height,
//               output_width,
//               rheight,
//               rwidth,
//               align_corners,
//               idata,
//               odata,
//               channels,
//               grad_output.numel(),
//               grad_input.numel());
//       C10_CUDA_KERNEL_LAUNCH_CHECK();
//     } else {
//       using accscalar_t = at::acc_type<scalar_t, true>;

//       const size_t num_kernels = nbatch * channels * input_height * input_width;
//       const int num_threads = 256;
//       // This is needed for non-contiguous tensors.
//       Tensor grad_input_c = grad_input.is_contiguous() ? grad_input : at::zeros(grad_input.sizes(), grad_input.options());
//       Tensor grad_output = grad_output_.contiguous();

//       auto idata = grad_input_c.data_ptr<scalar_t>();
//       auto odata = grad_output.data_ptr<scalar_t>();

//       const accscalar_t rheight = area_pixel_compute_scale<accscalar_t>(
//           input_height, output_height, align_corners, scales_h);
//       const accscalar_t rwidth = area_pixel_compute_scale<accscalar_t>(
//           input_width, output_width, align_corners, scales_w);
//       upsample_bilinear2d_backward_out_frame<scalar_t, accscalar_t>
//           <<<ceil_div(num_kernels, static_cast<size_t>(num_threads)),
//              num_threads,
//              0,
//              stream>>>(
//               nbatch * channels,
//               input_height,
//               input_width,
//               output_height,
//               output_width,
//               rheight,
//               rwidth,
//               align_corners,
//               idata,
//               odata);
//       C10_CUDA_KERNEL_LAUNCH_CHECK();

//       if (!grad_input.is_contiguous()) {
//           grad_input.copy_(grad_input_c);
//       }
//     }
//   });
// }


// Tensor upsample_bilinear_2d_cuda(
//     const Tensor& input,
//     IntArrayRef output_size,
//     bool align_corners,
//     c10::optional<double> scales_h,
//     c10::optional<double> scales_w) {
//   auto in_size = input.sizes();
//   Tensor output = Tensor();
//   if (input.is_contiguous(at::MemoryFormat::ChannelsLast)) {
//     output = at::empty(
//         {in_size[0], in_size[1], output_size[0], output_size[1]}, input.options().memory_format(at::MemoryFormat::ChannelsLast));
//   } else {
//     output = at::empty(
//         {in_size[0], in_size[1], output_size[0], output_size[1]}, input.options());
//   }

//   upsample_bilinear2d_out_cuda_template(
//     output, input, output_size, align_corners, scales_h, scales_w
//   );
//   return output;
// }


// Tensor upsample_bilinear_2d_backward_cuda(
//     const Tensor& grad_output,
//     IntArrayRef output_size,
//     IntArrayRef input_size,
//     bool align_corners,
//     c10::optional<double> scales_h,
//     c10::optional<double> scales_w) {
//   Tensor grad_input = Tensor();
//   if (grad_output.is_contiguous(at::MemoryFormat::ChannelsLast)) {
//     grad_input = at::zeros(input_size, grad_output.options().memory_format(at::MemoryFormat::ChannelsLast));
//   } else {
//     grad_input = at::zeros(input_size, grad_output.options());
//   }

//   upsample_bilinear2d_backward_out_cuda_template(
//     grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w
//   );
//   return grad_input;
// }


// Tensor upsample_bilinear_2d_forward(
//     const Tensor& input,
//     c10::OptionalIntArrayRef output_size,
//     bool align_corners,
//     c10::optional<ArrayRef<double>> scales_factors) {
//   auto osize = compute_output_size(input.sizes(), output_size, scales_factors);
//   auto scales_h = get_scale_value(scales_factors, 0);
//   auto scales_w = get_scale_value(scales_factors, 1);
//   return upsample_bilinear_2d_cuda(input, osize, align_corners, scales_h, scales_w);
// }

// Tensor upsample_bilinear_2d_backward(
//     const Tensor& grad_output,
//     c10::OptionalIntArrayRef output_size,
//     IntArrayRef input_size,
//     bool align_corners,
//     c10::optional<ArrayRef<double>> scales_factors) {
//   auto osize = compute_output_size(input_size, output_size, scales_factors);
//   auto scales_h = get_scale_value(scales_factors, 0);
//   auto scales_w = get_scale_value(scales_factors, 1);
//   return upsample_bilinear_2d_backward_cuda(grad_output, osize, input_size, align_corners, scales_h, scales_w);
// }



// Each thread writes one output pixel.
// Grid layout: (ceil(W_out/BLOCK_X), ceil(H_out/BLOCK_Y), N*C)
template <typename T>
__global__ void bilinear_upsample_kernel(
    const T* __restrict__ input,   // [N, C, H_in,  W_in ]
    T*       __restrict__ output,  // [N, C, H_out, W_out]
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const float scale_y,   // H_in / H_out
    const float scale_x    // W_in / W_out
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;   // flattened N*C index

    if (w_out >= W_out || h_out >= H_out) return;

    
    // 计算得到原来的中心坐标  -0.5？
    float src_y = (h_out + 0.5f) * scale_y - 0.5f;
    float src_x = (w_out + 0.5f) * scale_x - 0.5f;

    // 得到原来的起始点坐标 左上角
    int y0 = static_cast<int>(floorf(src_y));
    int x0 = static_cast<int>(floorf(src_x));
    int y1 = y0 + 1;
    int x1 = x0 + 1;

    y0 = std::max(0, std::min(y0, H_in - 1));
    y1 = std::max(0, std::min(y1, H_in - 1));
    x0 = std::max(0, std::min(x0, W_in - 1));
    x1 = std::max(0, std::min(x1, W_in - 1));

    // Fractional part used as interpolation weight toward the larger index
    float alpha = src_y - floorf(src_y);  // weight for y1
    float beta  = src_x - floorf(src_x);  // weight for x1

    const T* in_nc = input + (long)nc * H_in * W_in;
    float I00 = static_cast<float>(in_nc[y0 * W_in + x0]);
    float I01 = static_cast<float>(in_nc[y0 * W_in + x1]);
    float I10 = static_cast<float>(in_nc[y1 * W_in + x0]);
    float I11 = static_cast<float>(in_nc[y1 * W_in + x1]);

    float val = (1.0f - alpha) * (1.0f - beta) * I00
              + (1.0f - alpha) * beta  * I01
              + alpha  * (1.0f - beta) * I10
              + alpha  * beta  * I11;

    output[(long)nc * H_out * W_out + h_out * W_out + w_out] = static_cast<T>(val);
}


template <typename T, int vec>
__global__ void bilinear_upsample_backward_kernel_fp32(
    const T*   __restrict__ grad_output,    // [N, C, H_out, W_out]
    float*     __restrict__ grad_input_f32, // [N, C, H_in,  W_in ] 已预置零
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const float scale_y,   // H_in / H_out
    const float scale_x    // W_in / W_out
) {
    int w_out = (blockIdx.x * blockDim.x + threadIdx.x)*vec;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    // 与前向完全相同的坐标映射
    float src_y = (h_out + 0.5f) * scale_y - 0.5f;
    //float src_x = (w_out + 0.5f) * scale_x - 0.5f;
    float src_x[vec];
    for(int i = 0; i < vec; i++) {
        src_x[i] = (w_out + 0.5f + i) * scale_x - 0.5f;
    }

    int y0 = static_cast<int>(floorf(src_y));
    //int x0 = static_cast<int>(floorf(src_x));
    int x0[vec];
    int x1[vec];
    float beta[vec];
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        x0[i] = static_cast<int>(floorf(src_x[i]));
        x1[i] = x0[i] + 1;
        x0[i] = std::max(0, std::min(x0[i], W_in - 1));
        x1[i] = std::max(0, std::min(x1[i], W_in - 1));
        beta[i] = src_x[i] - x0[i]; 
    }
    int y1 = y0 + 1;

    y0 = std::max(0, std::min(y0, H_in - 1));
    y1 = std::max(0, std::min(y1, H_in - 1));

    float alpha = src_y - floorf(src_y);
    //float beta  = src_x - floorf(src_x);
    

    // 读取 grad_output（转为 float 参与后续计算）
    //float g = static_cast<float>(grad_output[nc * H_out * W_out + h_out * W_out + w_out]);
    using vec_grad = __attribute__((__vector_size__(vec * sizeof(float)))) float;
    vec_grad g_vec = *(vec_grad*)(grad_output + nc * H_out * W_out + h_out * W_out + w_out);

    // 向 4 个输入像素散射（scatter-add），使用 float32 atomicAdd 保证精度
    float* gin = grad_input_f32 + nc * H_in * W_in;
    #ifdef __gfx936__

        if constexpr (1) {
            #pragma unroll
            for(int i = 0; i < vec; i++) {
                float g = static_cast<float>(g_vec[i]);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x0[i]), (1.0f - alpha) * (1.0f - beta[i]) * g);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x1[i]), (1.0f - alpha) *          beta[i]  * g);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x0[i]),          alpha  * (1.0f - beta[i]) * g);
                __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x1[i]),          alpha  *          beta[i]  * g);
            }
        } else {
            // x4的原子加有问题 x1[0] x1[1] x1[2] x1[3] 不一定是连续的值排列
            vec_grad xy00, xy01, xy10, xy11;
            #pragma unroll
            for(int i = 0; i < vec; i++) {
                float g = static_cast<float>(g_vec[i]);
                xy00[i] = (1.0f - alpha) * (1.0f - beta[i]) * g;
                xy01[i] = (1.0f - alpha) *          beta[i]  * g;
                xy10[i] =          alpha  * (1.0f - beta[i]) * g;
                xy11[i] =          alpha  *          beta[i]  * g;
            }
        
        
            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y0 * W_in + x0[0]), xy00);
            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y1 * W_in + x0[0]), xy10);

            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y1 * W_in + x1[0]), xy11);
            __builtin_hcu_global_atomic_fadd_f32_x4((gin + y0 * W_in + x1[0]), xy01);
        }
    #else
        #pragma unroll
        for(int i = 0; i < vec; i++) {
            float g = static_cast<float>(g_vec[i]);
            atomicAdd(&gin[y0 * W_in + x0[i]], (1.0f - alpha) * (1.0f - beta[i]) * g);
            atomicAdd(&gin[y0 * W_in + x1[i]], (1.0f - alpha) *          beta[i]  * g);
            atomicAdd(&gin[y1 * W_in + x0[i]],          alpha  * (1.0f - beta[i]) * g);
            atomicAdd(&gin[y1 * W_in + x1[i]],          alpha  *          beta[i]  * g);
        }
    #endif
}

template <typename T, int vec>
__global__ void bilinear_upsample_backward_kernel_fp16(
    const T*   __restrict__ grad_output,    // [N, C, H_out, W_out]
    float*     __restrict__ grad_input_f32, // [N, C, H_in,  W_in ] 已预置零
    const int H_in,
    const int W_in,
    const int H_out,
    const int W_out,
    const float scale_y,   // H_in / H_out
    const float scale_x    // W_in / W_out
) {
    int w_out = (blockIdx.x * blockDim.x + threadIdx.x) * vec;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int nc    = blockIdx.z;

    if (w_out >= W_out || h_out >= H_out) return;

    // 与前向完全相同的坐标映射
    float src_y = (h_out + 0.5f) * scale_y - 0.5f;
    //float src_x = (w_out + 0.5f) * scale_x - 0.5f;
    float src_x[vec];
    for(int i = 0; i < vec; i++) {
        src_x[i] = (w_out + 0.5f + i) * scale_x - 0.5f;
    }

    int y0 = static_cast<int>(floorf(src_y));
    //int x0 = static_cast<int>(floorf(src_x));
    int x0[vec];
    int x1[vec];
    float beta[vec];
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        x0[i] = static_cast<int>(floorf(src_x[i]));
        x1[i] = x0[i] + 1;
        x0[i] = std::max(0, std::min(x0[i], W_in - 1));
        x1[i] = std::max(0, std::min(x1[i], W_in - 1));
        beta[i] = src_x[i] - x0[i]; 
    }
    int y1 = y0 + 1;

    y0 = std::max(0, std::min(y0, H_in - 1));
    y1 = std::max(0, std::min(y1, H_in - 1));

    float alpha = src_y - floorf(src_y);
    //float beta  = src_x - floorf(src_x);
    

    // 读取 grad_output（转为 float 参与后续计算）
    //float g = static_cast<float>(grad_output[nc * H_out * W_out + h_out * W_out + w_out]);
    using vec_grad = __attribute__((__vector_size__(vec * sizeof(_Float16)))) _Float16;
    vec_grad g_vec = *(vec_grad*)(grad_output + nc * H_out * W_out + h_out * W_out + w_out);

    // 向 4 个输入像素散射（scatter-add），使用 float32 atomicAdd 保证精度
    float* gin = grad_input_f32 + nc * H_in * W_in;
    #ifdef __gfx936__
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        float g = static_cast<float>(g_vec[i]);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x0[i]), (1.0f - alpha) * (1.0f - beta[i]) * g);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y0 * W_in + x1[i]), (1.0f - alpha) *          beta[i]  * g);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x0[i]),          alpha  * (1.0f - beta[i]) * g);
        __builtin_amdgcn_global_atomic_fadd_f32((float*)(gin + y1 * W_in + x1[i]),          alpha  *          beta[i]  * g);
    }
    #else
    #pragma unroll
    for(int i = 0; i < vec; i++) {
        float g = static_cast<float>(g_vec[i]);
        atomicAdd(&gin[y0 * W_in + x0[i]], (1.0f - alpha) * (1.0f - beta[i]) * g);
        atomicAdd(&gin[y0 * W_in + x1[i]], (1.0f - alpha) *          beta[i]  * g);
        atomicAdd(&gin[y1 * W_in + x0[i]],          alpha  * (1.0f - beta[i]) * g);
        atomicAdd(&gin[y1 * W_in + x1[i]],          alpha  *          beta[i]  * g);
    }
    #endif
}

// ============================================================
//  Forward host wrapper
//  支持 dtype: float32 / float16 / bfloat16
// ============================================================
torch::Tensor bilinear_upsample(
    const torch::Tensor&       x,
    const std::vector<int64_t>& output_size
) {
    TORCH_CHECK(x.dim() == 4,
        "bilinear_upsample: input must be 4-D [N, C, H, W], got shape ", x.sizes());
    TORCH_CHECK(output_size.size() == 2,
        "bilinear_upsample: output_size must have exactly 2 elements [H_out, W_out]");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float   ||
                x.scalar_type() == at::ScalarType::Half    ||
                x.scalar_type() == at::ScalarType::BFloat16,
        "bilinear_upsample: only float32 / float16 / bfloat16 are supported, got ",
        x.scalar_type());

    auto x_contig = x.contiguous();
    auto stream   = at::cuda::getCurrentCUDAStream();

    const int N    = static_cast<int>(x_contig.size(0));
    const int C    = static_cast<int>(x_contig.size(1));
    const int H_in = static_cast<int>(x_contig.size(2));
    const int W_in = static_cast<int>(x_contig.size(3));
    const int H_out = static_cast<int>(output_size[0]);
    const int W_out = static_cast<int>(output_size[1]);

    TORCH_CHECK(H_out > 0 && W_out > 0,
        "bilinear_upsample: output_size must be positive, got [", H_out, ", ", W_out, "]");

    const float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
    const float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

    auto output = torch::empty({N, C, H_out, W_out}, x_contig.options());

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (W_out + BLOCK_X - 1) / BLOCK_X,
        (H_out + BLOCK_Y - 1) / BLOCK_Y,
        N * C
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_contig.scalar_type(), "bilinear_upsample",
        [&]() {
            bilinear_upsample_kernel<scalar_t><<<grid, block, 0, stream>>>(
                x_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                H_in, W_in, H_out, W_out,
                scale_y, scale_x
            );
        }
    );

    return output;
}

// ============================================================
//  Backward host wrapper
//  支持 dtype: float32 / float16 / bfloat16
//
//  Parameters:
//    grad_output - 4-D tensor [N, C, H_out, W_out]
//    input_size  - 原始输入形状 {N, C, H_in, W_in}
//
//  Returns:
//    grad_input  - 4-D tensor [N, C, H_in, W_in]，dtype 与 grad_output 相同
// ============================================================
torch::Tensor bilinear_upsample_backward(
    const torch::Tensor&        grad_output,
    const std::vector<int64_t>& input_size
) {
    TORCH_CHECK(grad_output.dim() == 4,
        "bilinear_upsample_backward: grad_output must be 4-D, got shape ", grad_output.sizes());
    TORCH_CHECK(input_size.size() == 4,
        "bilinear_upsample_backward: input_size must have 4 elements [N, C, H_in, W_in]");
    TORCH_CHECK(grad_output.scalar_type() == at::ScalarType::Float   ||
                grad_output.scalar_type() == at::ScalarType::Half    ||
                grad_output.scalar_type() == at::ScalarType::BFloat16,
        "bilinear_upsample_backward: only float32 / float16 / bfloat16 are supported, got ",
        grad_output.scalar_type());

    auto go_contig = grad_output.contiguous();
    auto stream    = at::cuda::getCurrentCUDAStream();

    const int N     = static_cast<int>(input_size[0]);
    const int C     = static_cast<int>(input_size[1]);
    const int H_in  = static_cast<int>(input_size[2]);
    const int W_in  = static_cast<int>(input_size[3]);
    const int H_out = static_cast<int>(go_contig.size(2));
    const int W_out = static_cast<int>(go_contig.size(3));

    const float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
    const float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

    
    constexpr int BLOCK_Y = 8;
    constexpr int BLOCK_X = 64;
    constexpr int vec = 2;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(
        (W_out + BLOCK_X*vec - 1) / (BLOCK_X * vec),
        (H_out + BLOCK_Y - 1) / BLOCK_Y,
        N * C
    );
    

    if (go_contig.scalar_type() == at::ScalarType::Float) {
        // fp32 路径：grad_input 本身是 float32，直接传指针，省去中间缓冲区
        auto grad_input = torch::zeros({N, C, H_in, W_in}, go_contig.options());
        bilinear_upsample_backward_kernel_fp32<float, vec><<<grid, block, 0, stream>>>(
            go_contig.data_ptr<float>(),
            grad_input.data_ptr<float>(),
            H_in, W_in, H_out, W_out,
            scale_y, scale_x
        );
        return grad_input;
    }

    // fp16 / bf16 路径：先向 float32 缓冲区累加，再 cast 回原始 dtype
    auto grad_input_fp32 = torch::zeros(
        {N, C, H_in, W_in},
        go_contig.options().dtype(torch::kFloat32));
    
    if(H_in == 16 && W_in == 31) {
        constexpr int BLOCK_Y1 = 32;
        constexpr int BLOCK_X1 = 32;
        constexpr int vec1 = 4;
        dim3 block1(BLOCK_X1, BLOCK_Y1);
        dim3 grid1(
            (W_out + BLOCK_X1*vec1 - 1) / (BLOCK_X1 * vec1),
            (H_out + BLOCK_Y1 - 1) / BLOCK_Y1,
            N * C
        );
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            go_contig.scalar_type(), "bilinear_upsample_backward",
            [&]() {
                bilinear_upsample_backward_kernel_fp16<scalar_t, vec1><<<grid1, block1, 0, stream>>>(
                    go_contig.data_ptr<scalar_t>(),
                    grad_input_fp32.data_ptr<float>(),
                    H_in, W_in, H_out, W_out,
                    scale_y, scale_x
                );
            }
        );
    }
    else{
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            go_contig.scalar_type(), "bilinear_upsample_backward",
            [&]() {
                bilinear_upsample_backward_kernel_fp16<scalar_t, vec><<<grid, block, 0, stream>>>(
                    go_contig.data_ptr<scalar_t>(),
                    grad_input_fp32.data_ptr<float>(),
                    H_in, W_in, H_out, W_out,
                    scale_y, scale_x
                );
            }
        );
    }

    return grad_input_fp32.to(grad_output.dtype());
}


Tensor upsample_bilinear_2d_forward(
    const Tensor& input,
    c10::OptionalIntArrayRef output_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scales_factors) {
  auto osize = compute_output_size(input.sizes(), output_size, scales_factors);
  std::vector<int64_t> output_size_v(osize.begin(), osize.end());
  return bilinear_upsample(input, output_size_v);
}

Tensor upsample_bilinear_2d_backward(
    const Tensor& grad_output,
    c10::OptionalIntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<ArrayRef<double>> scales_factors) {
  std::vector<int64_t> input_size_v(input_size.begin(), input_size.end());
  return bilinear_upsample_backward(grad_output, input_size_v);
}

}
}
