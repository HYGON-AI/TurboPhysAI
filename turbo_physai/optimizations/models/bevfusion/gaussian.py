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

"""Device-side Gaussian heatmap helpers used by BEVFusion."""

from ...common.mmdet3d.gaussian import gaussian_2d


def _compiled_replacement(replacement):
    """Compile one optimized helper with the mode used by the reference tree."""

    import functools
    import os
    import torch

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return replacement
    compiled = torch.compile(
        replacement,
        mode="max-autotune-no-cudagraphs",
        fullgraph=False,
        dynamic=False,
    )

    @functools.wraps(replacement)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped


def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    import torch

    diameter = 2 * radius + 1
    gaussian = gaussian_2d(
        (diameter, diameter),
        sigma=diameter / 6,
        device=heatmap.device,
        dtype=heatmap.dtype,
    )
    offsets = torch.arange(
        -radius, radius + 1, device=heatmap.device, dtype=center.dtype
    )
    ys = center[1] + offsets[:, None]
    xs = center[0] + offsets[None, :]
    height, width = heatmap.shape
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    indices = ys.clamp(0, height - 1) * width + xs.clamp(0, width - 1)
    values = torch.where(valid, gaussian * k, torch.finfo(heatmap.dtype).min)
    heatmap.flatten().scatter_reduce_(
        0,
        indices.flatten().long(),
        values.flatten(),
        reduce="amax",
        include_self=True,
    )
    return heatmap


def gaussian_radius(det_size, min_overlap=0.5):
    import torch

    height, width = det_size
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + torch.sqrt(b1**2 - 4 * a1 * c1)) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 + torch.sqrt(b2**2 - 4 * a2 * c2)) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + torch.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return torch.stack((r1, r2, r3)).min(0).values


def compiled_draw_heatmap_gaussian_wrapper(original, options):
    """Replace the baseline helper with the compiled optimized implementation."""

    del original, options
    return _compiled_replacement(draw_heatmap_gaussian)


def compiled_gaussian_radius_wrapper(original, options):
    """Replace the baseline radius helper with its compiled implementation."""

    del original, options
    return _compiled_replacement(gaussian_radius)
