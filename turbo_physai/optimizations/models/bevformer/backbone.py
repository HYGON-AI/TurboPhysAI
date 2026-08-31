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

"""BEVFormer backbone extraction replacement."""

import os

import torch


def extract_img_feat(self, img, img_metas, len_queue=None):
    """Extract image features with channels-last backbone inputs."""

    del img_metas
    if img is None:
        return None
    batch_size = img.size(0)
    if img.dim() == 5 and img.size(0) == 1:
        img = img.squeeze(0)
    elif img.dim() == 5:
        batch_size, cameras, channels, height, width = img.size()
        img = img.reshape(batch_size * cameras, channels, height, width)
    if self.use_grid_mask:
        img = self.grid_mask(img)
    img = img.to(memory_format=torch.channels_last)
    img_feats = self.img_backbone(img)
    if isinstance(img_feats, dict):
        img_feats = list(img_feats.values())
    if self.with_img_neck:
        img_feats = self.img_neck(img_feats)

    reshaped = []
    for img_feat in img_feats:
        batch_cameras, channels, height, width = img_feat.size()
        if len_queue is not None:
            reshaped.append(
                img_feat.view(
                    int(batch_size / len_queue),
                    len_queue,
                    int(batch_cameras / batch_size),
                    channels,
                    height,
                    width,
                )
            )
        else:
            reshaped.append(
                img_feat.view(
                    batch_size,
                    int(batch_cameras / batch_size),
                    channels,
                    height,
                    width,
                )
            )
    return reshaped


def compiled_extract_img_feat(original, options):
    """Build the fixed BEVFormer backbone replacement for WrapperHandler."""

    del original
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return extract_img_feat
    mode = options.get("mode", "max-autotune-no-cudagraphs")
    return torch.compile(extract_img_feat, mode=mode)
