// Derived from PyTorch: aten/src/ATen/native/UpSample.h
// PyTorch v1.13.1, commit 49444c3e546bf240bed24a101e747422d1f8a0ee.
// SPDX-License-Identifier: BSD-3-Clause
// Copyright 2026 Hygon Information Technology Co., Ltd.
// Modified by Hygon.

#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAContext.h>

namespace {
template <typename scalar_t>
inline c10::optional<scalar_t> get_scale_value(c10::optional<c10::ArrayRef<scalar_t>> scales, int idx) {
  if (!scales) {
    return c10::nullopt;
  }
  return scales->at(idx);
}

inline c10::optional<int64_t> get_scale_value(c10::OptionalIntArrayRef scales, int idx) {
  if (!scales) {
    return c10::nullopt;
  }
  return scales->at(idx);
}
}
