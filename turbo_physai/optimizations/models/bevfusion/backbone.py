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

"""BEVFusion feature extraction replacements."""

import os


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {
        "1", "true", "on", "yes"
    }


def extract_camera_features(
    self,
    x,
    points,
    radar_points,
    camera2ego,
    lidar2ego,
    lidar2camera,
    lidar2image,
    camera_intrinsics,
    camera2lidar,
    img_aug_matrix,
    lidar_aug_matrix,
    img_metas,
    gt_depths=None,
):
    """Run the camera branch with the reference channels-last input policy."""

    import torch

    batch_size, cameras, channels, height, width = x.size()
    x = x.view(batch_size * cameras, channels, height, width)
    if _env_flag("MMDET3D_CHANNELS_LAST"):
        x = x.contiguous(memory_format=torch.channels_last)

    x = self.encoders["camera"]["backbone"](x)
    x = self.encoders["camera"]["neck"](x)

    if not isinstance(x, torch.Tensor):
        x = x[0]

    batch_cameras, channels, height, width = x.size()
    x = x.view(batch_size, int(batch_cameras / batch_size), channels, height, width)

    x = self.encoders["camera"]["vtransform"](
        x,
        points,
        radar_points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        depth_loss=self.use_depth_loss,
        gt_depths=gt_depths,
    )
    return x


def extract_features(self, x, sensor):
    """Use the input list length instead of synchronizing on GPU coordinates."""

    batch_size = len(x)
    feats, coords, sizes = self.voxelize(x, sensor)
    return self.encoders[sensor]["backbone"](
        feats, coords, batch_size, sizes=sizes
    )
