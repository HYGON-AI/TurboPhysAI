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
Muon 的 update gather 有两条路径：

1. gather_updates：all_gather_into_tensor + movedim。每张卡都拿到完整的[N, M, K]，适合每个 rank 都要算全部矩阵的场景。
2. gather_owned_updates：packed all_to_all_single。只把 本rank负责的矩阵、在其他rank的那几行 搬过来。muon_owner_ns 之后每张卡只算 1/world_size 的矩阵，
用 all_gather 会多传输 world_size 倍的无效数据。
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import Tensor

from .muon_owner_ns import partition_range, shard_range
from .muon_row_pack import unpack_rows

def gather_updates(stacked_local: Tensor, *, global_size: int, shard_dim: int,
                   world_size: int, rank: int, group) -> Tensor:
    """[N, 局部 M, K] -> [N, M, K]：movedim 后使用 all_gather_into_tensor"""
    gather_dim = shard_dim + 1
    if world_size == 1:
        return stacked_local

    sizes = [shard_range(global_size, world_size, r)[1] for r in range(world_size)]
    max_size = max(sizes)
    local_size = sizes[rank]
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

    if min(sizes) != max_size:
        gathered = torch.cat(
            [
                gathered.narrow(0, r * max_size, sizes[r])
                for r in range(world_size)
            ],
            dim=0,
        )
    return gathered.movedim(0, gather_dim)


def gather_owned_updates(
    stacked_local: Tensor,
    *,
    matrix_count: int,
    global_size: int,
    shard_dim: int,
    own_count: int,
    world_size: int,
    group,
) -> Tensor:
    if world_size == 1:
        return stacked_local
    if stacked_local.shape[0] != matrix_count:
        raise ValueError(
            f"gather_owned_updates expects {matrix_count} matrices, "
            f"got {stacked_local.shape[0]}"
        )
    if not stacked_local.is_contiguous():
        stacked_local = stacked_local.contiguous()
    if own_count <= 0:
        raise ValueError("gather_owned_updates requires at least one owned matrix")

    axis = shard_dim + 1
    send_counts = [partition_range(matrix_count, world_size, d)[1] for d in range(world_size)]
    if sum(send_counts) != matrix_count:
        raise ValueError("matrix partition does not cover every matrix")

    row_ranges = [shard_range(global_size, world_size, s) for s in range(world_size)]
    out_shape = list(stacked_local.shape[1:])
    out_shape[shard_dim] = global_size
    piece_shape = list(stacked_local.shape[1:])
    recv_sizes = []
    for _, rows in row_ranges:
        shape = list(piece_shape)
        shape[shard_dim] = rows
        recv_sizes.append(own_count * math.prod(shape))
    per_matrix = stacked_local.numel() // matrix_count
    send_sizes = [count * per_matrix for count in send_counts]

    if sum(send_sizes) != stacked_local.numel():
        raise ValueError("packed send buffer does not cover the local shard")
    if sum(recv_sizes) != own_count * math.prod(out_shape):
        raise ValueError("packed recv buffer does not cover the owned matrices")
    if 0 in send_sizes or 0 in recv_sizes:
        raise RuntimeError(
            f"zero-size all_to_all is not supported on this stack: "
            f"send={send_sizes} recv={recv_sizes}"
        )

    recv = stacked_local.new_empty(sum(recv_sizes))
    dist.all_to_all_single(
        recv,
        stacked_local.reshape(-1),
        output_split_sizes=recv_sizes,
        input_split_sizes=send_sizes,
        group=group,
    )

    out = stacked_local.new_empty((own_count,) + tuple(out_shape))
    if axis == 1 and unpack_rows(recv, out, count=own_count, world_size=world_size):
        return out
    offset = 0
    for source, (start, rows) in enumerate(row_ranges):
        numel = recv_sizes[source]
        if numel:
            shape = list(piece_shape)
            shape[shard_dim] = rows
            piece = recv.narrow(0, offset, numel).view((own_count,) + tuple(shape))
            index = [slice(None)] * out.ndim
            index[axis] = slice(start, start + rows)
            out[tuple(index)].copy_(piece)
        offset += numel
    return out
