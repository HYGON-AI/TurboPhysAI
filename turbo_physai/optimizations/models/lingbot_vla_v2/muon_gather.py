# Copyright 2026 Robbyant Team and/or its affiliates
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

"""
Muon 的 update gather：使用all_gather_into_tensor + movedim，避免拉到一个列表里后当场cat，省一次读写跟显存。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor

def gather_updates(stacked_local: Tensor, *, global_size: int, shard_dim: int,
                   world_size: int, rank: int, group) -> Tensor:
    """[N, 局部 M, K] -> [N, M, K]：movedim 后使用 all_gather_into_tensor"""
    gather_dim = shard_dim + 1
    floor, remainder = divmod(global_size, world_size)
    local_size = floor + (rank < remainder)
    if world_size == 1:
        return stacked_local

    max_size = floor + bool(remainder)
    if local_size != max_size:
        shape = list(stacked_local.shape)
        shape[gather_dim] = max_size
        padded = stacked_local.new_zeros(shape)
        padded.narrow(gather_dim, 0, local_size).copy_(stacked_local)
    else:
        padded = stacked_local

    packed = padded.movedim(gather_dim, 0).contiguous()
    gathered = packed.new_empty((world_size * max_size,) + tuple(packed.shape[1:]))
    dist.all_gather_into_tensor(gathered, packed, group=group)

    if remainder:
        gathered = torch.cat(
            [
                gathered.narrow(0, r * max_size, floor + (r < remainder))
                for r in range(world_size)
            ],
            dim=0,
        )
    return gathered.movedim(0, gather_dim)
