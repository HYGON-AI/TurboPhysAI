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

#include <torch/extension.h>
#include "voxelization.h"

namespace voxelization {

void bind_voxelization(py::module_ &m) {
  m.def("hard_voxelize", &hard_voxelize, "hard voxelize");
  m.def("dynamic_voxelize", &dynamic_voxelize, "dynamic voxelization");
  m.def("dynamic_point_to_voxel_forward", &dynamic_point_to_voxel_forward, "dynamic point to voxel forward");
  m.def("dynamic_point_to_voxel_backward", &dynamic_point_to_voxel_backward, "dynamic point to voxel backward");
}

} // namespace voxelization
