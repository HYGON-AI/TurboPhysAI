# Copyright 2018-2019 OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""Reusable device-side Gaussian kernel implementation."""


def gaussian_2d(shape, sigma=1, device=None, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.float32
    middle_y, middle_x = [(size - 1.0) / 2.0 for size in shape]
    y = torch.arange(
        -middle_y, middle_y + 1, device=device, dtype=dtype
    ).unsqueeze(1)
    x = torch.arange(
        -middle_x, middle_x + 1, device=device, dtype=dtype
    ).unsqueeze(0)
    heatmap = torch.exp(-(x.square() + y.square()) / (2 * sigma * sigma))
    return torch.where(
        heatmap < torch.finfo(heatmap.dtype).eps * heatmap.max(),
        0,
        heatmap,
    )
