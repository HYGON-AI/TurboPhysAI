# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""将独立的矩阵放在不同分片上算，之前是所有矩阵在rank0-7上都算一次，相当于多算了7/8，现在rank0算0-3矩阵，rank1算4-7矩阵，减少重复计算"""
import math

import torch
import torch.distributed as dist


def partition_range(size, world_size, rank):
    if size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid partition")
    q, r = divmod(size, world_size)
    return rank * q + min(rank, r), q + int(rank < r)


def pack_result(owned, *, matrix_count, world_size, rank, shard_dim):
    """owner 的 [owned_N,M,K] → 按目标 rank 排列的一维发送缓冲。"""
    if owned.ndim != 3 or shard_dim not in (0, 1):
        raise ValueError("owner NS requires batched 2D matrices")
    _, count = partition_range(matrix_count, world_size, rank)
    if owned.shape[0] != count:
        raise ValueError("owner matrix count mismatch")
    axis = shard_dim + 1
    if owned.shape[axis] % world_size:
        raise ValueError("owner NS v1 requires evenly divisible storage shards")
    sizes = []
    pieces = []
    for dest in range(world_size):
        start, length = partition_range(owned.shape[axis], world_size, dest)
        piece = owned.narrow(axis, start, length).contiguous().reshape(-1)
        pieces.append(piece)
        sizes.append(piece.numel())
    return torch.cat(pieces), sizes


def exchange_result(owned, *, matrix_count, world_size, rank, shard_dim, group):
    """无求和的 all-to-all，接收后恢复原矩阵顺序和本地 shard。"""
    send, send_sizes = pack_result(
        owned, matrix_count=matrix_count, world_size=world_size,
        rank=rank, shard_dim=shard_dim,
    )
    local_shape = list(owned.shape[1:])
    _, local_shape[shard_dim] = partition_range(local_shape[shard_dim], world_size, rank)
    elements = math.prod(local_shape)
    recv_sizes = [partition_range(matrix_count, world_size, src)[1] * elements
                  for src in range(world_size)]
    recv = owned.new_empty(sum(recv_sizes))
    # 显式 split sizes 支持不均分矩阵/存储分片。后端错误直接抛出，不能单 rank 回退。
    dist.all_to_all_single(recv, send, output_split_sizes=recv_sizes,
                           input_split_sizes=send_sizes, group=group)
    return recv.reshape(matrix_count, *local_shape)
