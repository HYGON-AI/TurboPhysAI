# Copyright (c) 2022 Hust Vision Lab
# Derived from hustvl/MapTR at e03f097abef19e1ba3fed5f471a8d80fbfa0a064.
# Upstream: projects/mmdet3d_plugin/models/utils/grid_mask.py
# Modified by Hygon; upstream license: MIT.
# See THIRD_PARTY_NOTICES.md and third_party/licenses/MapTR-MIT.txt.
# Existing OpenMMLab Apache-2.0 notices below are retained.
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

"""Dynamo-safe ``GridMask.forward`` replacement.

The official implementation builds the mask with ``numpy``/``PIL`` and then
calls ``torch.from_numpy(mask).to(x.dtype).cuda()``.  Two problems follow:

* ``np.asarray`` on a ``PIL`` image returns a read-only array.  ``torch.from_numpy``
  shares that buffer, so the resulting tensor can never be written to.  Dynamo
  and AOTAutograd guard on tensor storages, so the read-only flag turns into a
  graph break or a hard failure once surrounding code is compiled.
* ``.cuda()`` pins the mask to CUDA, which is wrong on a DCU/HIP build and
  forces an extra device transfer even when the input already lives there.

Keeping the official mask construction and fixing exactly those two points gives
a bit-identical mask while staying device agnostic, so ``rotate`` modes other
than the identity rotation keep working unchanged.  The function is additionally
handed to ``torch._dynamo.disable`` on first use, because the random/Python
control flow in this augmentation cannot be traced usefully.
"""

import numpy as np
import torch


def _grid_mask_forward(self, x):
    """Official mask construction with a writable, device-local mask tensor."""

    if np.random.rand() > self.prob or not self.training:
        return x

    n, c, h, w = x.size()
    flat = x.view(-1, h, w)
    expanded_h = int(1.5 * h)
    expanded_w = int(1.5 * w)
    distance = np.random.randint(2, h)
    self.l = min(max(int(distance * self.ratio + 0.5), 1), distance - 1)
    mask = np.ones((expanded_h, expanded_w), np.float32)
    start_h = np.random.randint(distance)
    start_w = np.random.randint(distance)
    if self.use_h:
        for index in range(expanded_h // distance):
            start = distance * index + start_h
            mask[start:min(start + self.l, expanded_h), :] *= 0
    if self.use_w:
        for index in range(expanded_w // distance):
            start = distance * index + start_w
            mask[:, start:min(start + self.l, expanded_w)] *= 0

    from PIL import Image

    rotated = Image.fromarray(np.uint8(mask)).rotate(
        np.random.randint(self.rotate))
    # ``.copy()`` detaches the read-only PIL buffer so ``torch.from_numpy``
    # yields a writable tensor that Dynamo can guard on.
    mask = np.asarray(rotated).copy()
    mask = mask[
        (expanded_h - h) // 2:(expanded_h - h) // 2 + h,
        (expanded_w - w) // 2:(expanded_w - w) // 2 + w,
    ]
    mask = torch.from_numpy(mask).to(device=x.device, dtype=x.dtype)
    if self.mode == 1:
        mask = 1 - mask
    mask = mask.expand_as(flat)
    if self.offset:
        offset = torch.from_numpy(
            2 * (np.random.rand(h, w) - 0.5)
        ).to(device=x.device, dtype=x.dtype)
        flat = flat * mask + offset * (1 - mask)
    else:
        flat = flat * mask
    return flat.view(n, c, h, w)


_guarded_forward = None


def _build_guarded_forward():
    """Install ``auto_fp16`` and, when available, the Dynamo opt-out."""

    from mmcv.runner import auto_fp16

    guarded = auto_fp16()(_grid_mask_forward)

    from .compat import dynamo_available

    if not dynamo_available():
        # A Torch build without ``torch._dynamo`` must still run the
        # augmentation instead of failing on the first forward call.
        return guarded

    import torch

    return torch._dynamo.disable(guarded)


def grid_mask_forward(self, x):
    """Entry point that installs the guards once, then delegates.

    The official ``forward`` carries ``@auto_fp16()``, so the replacement keeps
    the same decorator (applied lazily to avoid importing ``mmcv`` at module
    import time) and adds the Dynamo opt-out around it.
    """

    global _guarded_forward
    if _guarded_forward is None:
        _guarded_forward = _build_guarded_forward()
    return _guarded_forward(self, x)
