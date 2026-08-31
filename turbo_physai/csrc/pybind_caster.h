// Copyright 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

namespace pybind11::detail {
template <typename T>
struct type_caster<c10::optional<c10::ArrayRef<T>>> {
private:
    // vector 用于存储从 Python 传入的列表数据，确保在 C++ 端生命周期内有效
    std::vector<T> vec_;
public:
  using OptionalArrayRef = c10::optional<c10::ArrayRef<T>>;
  PYBIND11_TYPE_CASTER(OptionalArrayRef, _("c10::optional<c10::ArrayRef<T>>"));

  bool load(handle src, bool /* convert 不用传参 */) {
    if (src.is_none()) {
      value = c10::nullopt;
      return true;
    }

    try {
      vec_ = py::cast<std::vector<T>>(src);
      value = c10::make_optional<c10::ArrayRef<T>>(vec_);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle cast(const c10::optional<c10::ArrayRef<T>>& src,
                    return_value_policy policy, handle parent) {
    if (!src.has_value())
        return py::none().release();

    auto arr = src.value();
    std::vector<T> vec(arr.begin(), arr.end());
    return py::cast(vec, policy, parent).release();
  }
};

template <>
struct type_caster<c10::OptionalIntArrayRef> {
public:
  std::vector<int64_t> vec_;
public:
  using Type = c10::OptionalIntArrayRef;
  PYBIND11_TYPE_CASTER(Type, _("OptionalIntArrayRef"));

  bool load(handle src, bool /* convert */) {
    if (src.is_none()) {
      value = c10::nullopt;
      return true;
    }
    try {
      vec_ = py::cast<std::vector<int64_t>>(src);
      value = c10::ArrayRef<int64_t>(vec_);
      return true;
    } catch (...) {
      return false;
    }
  }

  static handle cast(const Type& src, return_value_policy policy, handle parent) {
    if (!src) return py::none().release();
    std::vector<int64_t> vec(src->begin(), src->end());
    return py::cast(vec, policy, parent).release();
  }
};

} // namespace pybind11::detail
