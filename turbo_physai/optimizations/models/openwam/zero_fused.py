# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, sparse-owner scaled packing; plans contain no gradient tensors."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

BLOCK = 4096


def tile_rows(spans):
    """Split (source offset, destination offset, length) spans into copy tiles."""
    return tuple(
        (src + offset, dst + offset, min(BLOCK, length - offset))
        for src, dst, length in spans
        for offset in range(0, length, BLOCK)
    )


def upload_metadata(rows, device):
    host = torch.tensor(rows, dtype=torch.int32).reshape(-1, 3)
    if device.type == "cuda":
        host = host.pin_memory()
    return host.to(device=device, non_blocking=device.type == "cuda")


@dataclass(frozen=True)
class Chunk:
    owner: int
    count: int
    direct: int | None  # Source offset, for the original one-span Reduce path.
    rows: tuple


def sparse_plan(ranges, world, max_elements):
    """Mirror reduce_by_owner's sparse branch and collective order exactly."""
    grouped = [[] for _ in range(world)]
    for offset, count, owner in ranges:
        if count:
            grouped[owner].append((offset, count))
    lengths = [sum(n for _, n in spans) for spans in grouped]
    if not sum(lengths) or max(lengths) * world <= 2 * sum(lengths):
        return None  # Balanced ReduceScatter retains the reference path.
    chunks = []
    for owner, spans in enumerate(grouped):
        if len(spans) == 1:
            offset, count = spans[0]
            chunks.append(Chunk(owner, count, offset, ()))
            continue
        for start in range(0, lengths[owner], max_elements):
            count = min(max_elements, lengths[owner] - start)
            cursor = 0
            copies = []
            for src, length in spans:
                left, right = max(start, cursor), min(start + count, cursor + length)
                if right > left:
                    copies.append((src + left - cursor, left - start, right - left))
                cursor += length
            chunks.append(Chunk(owner, count, None, tile_rows(copies)))
    return tuple(chunks)


class ScaledPack:
    def __init__(self):
        self.plans = {}
        self.metadata_bytes = 0

    def run(self, tensor, ranges, *, group, dtype, max_elements, staging_buffer):
        world = dist.get_world_size(group)
        # Power-of-two scaling preserves the existing float32 divide rounding.
        # All ranks use the same guards, so the reference collective order is kept.
        if (
            not tensor.is_cuda
            or tensor.dtype != torch.bfloat16
            or dtype != tensor.dtype
            or not tensor.is_contiguous()
            or tensor.numel() >= 2**31
            or world < 2
            or world & (world - 1)
            or max_elements < world
        ):
            return False
        from turbo_physai.optimizations.models.openwam.zero_kernels import (
            transfer_kernel,
        )

        rank = dist.get_rank(group)
        key = (tensor.device, tuple(ranges), world, max_elements)
        if key not in self.plans:
            if len(self.plans) >= 128:
                return False
            plan = sparse_plan(ranges, world, max_elements)
            if plan is None:
                self.plans[key] = None
            else:
                size = sum(len(c.rows) * 12 for c in plan)
                if self.metadata_bytes + size > 32 * 1024**2:
                    return False
                self.plans[key] = tuple(
                    (
                        c.owner,
                        c.count,
                        c.direct,
                        len(c.rows),
                        upload_metadata(c.rows, tensor.device),
                    )
                    for c in plan
                )
                self.metadata_bytes += size
        plan = self.plans[key]
        if plan is None:
            return False
        for owner, count, direct, tiles, metadata in plan:
            dst = dist.get_global_rank(group, owner) if group is not None else owner
            if direct is not None:
                view = tensor.narrow(0, direct, count)
                view.div_(world)
                work = dist.reduce(view, dst=dst, group=group, async_op=True)
                work.wait()
                continue
            packed = staging_buffer(tensor.device, dtype, max_elements).narrow(
                0, 0, count
            )
            transfer_kernel[(tiles,)](
                tensor,
                packed,
                metadata,
                1.0 / world,
                True,
                False,
                BLOCK=BLOCK,
                num_warps=4,
                enable_fp_fusion=False,
            )
            work = dist.reduce(packed, dst=dst, group=group, async_op=True)
            work.wait()  # Orders packed output and reuse on the reduction stream.
            if owner == rank:
                transfer_kernel[(tiles,)](
                    packed, tensor, metadata, 1.0, False, True, BLOCK=BLOCK, num_warps=4
                )
        return True
