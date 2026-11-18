# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""将独立的矩阵放在不同分片上算，之前是所有矩阵在rank0-7上都算一次，相当于多算了7/8，现在rank0算0-3矩阵，rank1算4-7矩阵，减少重复计算"""
import math

import torch
import torch.distributed as dist

from .muon_row_pack import pack_rows


def partition_range(size, world_size, rank):
    if size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid partition")
    q, r = divmod(size, world_size)
    return rank * q + min(rank, r), q + int(rank < r)


def shard_range(size, world_size, rank):
    if size < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid shard")
    chunk = -(-size // world_size)
    start = min(rank * chunk, size)
    return start, min(chunk, size - start)


def pack_result(owned, *, matrix_count, world_size, rank, shard_dim):
    if owned.ndim != 3 or shard_dim not in (0, 1):
        raise ValueError("owner NS requires batched 2D matrices")
    _, count = partition_range(matrix_count, world_size, rank)
    if owned.shape[0] != count:
        raise ValueError("owner matrix count mismatch")
    axis = shard_dim + 1
    ranges = [shard_range(owned.shape[axis], world_size, dest) for dest in range(world_size)]
    thickness = owned.numel() // (owned.shape[0] * owned.shape[axis])

    send = owned.new_empty(owned.numel())
    sizes = [count * length * thickness for _, length in ranges]
    if shard_dim == 0 and pack_rows(owned, send, count=count, world_size=world_size):
        return send, sizes
    offset = 0
    for start, length in ranges:
        piece = owned.narrow(axis, start, length)
        numel = piece.numel()
        send.narrow(0, offset, numel).view(piece.shape).copy_(piece)
        offset += numel
    if offset != send.numel():
        raise ValueError("packed send buffer does not cover the owned matrices")
    return send, sizes


def exchange_result(owned, *, matrix_count, world_size, rank, shard_dim, group):
    """无求和的 all-to-all，接收后恢复原矩阵顺序和本地 shard。"""
    send, send_sizes = pack_result(
        owned, matrix_count=matrix_count, world_size=world_size,
        rank=rank, shard_dim=shard_dim,
    )
    local_shape = list(owned.shape[1:])
    _, local_shape[shard_dim] = shard_range(local_shape[shard_dim], world_size, rank)
    elements = math.prod(local_shape)
    recv_sizes = [partition_range(matrix_count, world_size, src)[1] * elements
                  for src in range(world_size)]
    if 0 in send_sizes or 0 in recv_sizes:
        raise RuntimeError(
            f"zero-size all_to_all is not supported on this stack: "
            f"send={send_sizes} recv={recv_sizes}"
        )
    recv = owned.new_empty(sum(recv_sizes))
    # 显式 split sizes 支持不均分矩阵/存储分片。后端错误直接抛出，不能单 rank 回退。
    dist.all_to_all_single(recv, send, output_split_sizes=recv_sizes,
                           input_split_sizes=send_sizes, group=group)
    return recv.reshape(matrix_count, *local_shape)
