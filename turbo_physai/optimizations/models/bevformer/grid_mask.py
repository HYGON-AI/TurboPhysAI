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

"""Device-tensor GridMask replacement."""

import numpy as np
import torch


def grid_mask_forward(self, x):
    if np.random.rand() > self.prob or not self.training:
        return x
    n, channels, height, width = x.size()
    expanded_h = int(1.5 * height)
    expanded_w = int(1.5 * width)
    distance = np.random.randint(2, height)
    self.l = min(max(int(distance * self.ratio + 0.5), 1), distance - 1)
    start_h = np.random.randint(distance)
    start_w = np.random.randint(distance)
    if self.rotate == 1:
        top = (expanded_h - height) // 2
        left = (expanded_w - width) // 2
        rows = torch.arange(top, top + height, device=x.device)
        columns = torch.arange(left, left + width, device=x.device)
        row_mask = torch.ones(height, dtype=x.dtype, device=x.device)
        column_mask = torch.ones(width, dtype=x.dtype, device=x.device)
        if self.use_h:
            row_mask = ((rows - start_h).remainder(distance) >= self.l).to(x.dtype)
        if self.use_w:
            column_mask = ((columns - start_w).remainder(distance) >= self.l).to(x.dtype)
        mask = (row_mask[:, None] * column_mask[None, :]).view(
            1, 1, height, width
        )
        if self.mode == 1:
            mask = 1 - mask
        if self.offset:
            offset = 2 * (
                torch.rand(1, 1, height, width, dtype=x.dtype, device=x.device)
                - 0.5
            )
            return x * mask + offset * (1 - mask)
        return x * mask

    # Preserve the official implementation for rotated modes.
    from PIL import Image
    flat = x.view(-1, height, width)
    mask = np.ones((expanded_h, expanded_w), np.float32)
    if self.use_h:
        for index in range(expanded_h // distance):
            start = distance * index + start_h
            mask[start:min(start + self.l, expanded_h), :] *= 0
    if self.use_w:
        for index in range(expanded_w // distance):
            start = distance * index + start_w
            mask[:, start:min(start + self.l, expanded_w)] *= 0
    mask = np.asarray(Image.fromarray(np.uint8(mask)).rotate(
        np.random.randint(self.rotate)
    )).copy()
    mask = mask[
        (expanded_h - height) // 2:(expanded_h - height) // 2 + height,
        (expanded_w - width) // 2:(expanded_w - width) // 2 + width,
    ]
    mask = torch.from_numpy(mask).to(device=x.device, dtype=x.dtype)
    if self.mode == 1:
        mask = 1 - mask
    mask = mask.expand_as(flat)
    if self.offset:
        offset = torch.from_numpy(
            2 * (np.random.rand(height, width) - 0.5)
        ).to(device=x.device, dtype=x.dtype)
        flat = flat * mask + offset * (1 - mask)
    else:
        flat = flat * mask
    return flat.view(n, channels, height, width)
