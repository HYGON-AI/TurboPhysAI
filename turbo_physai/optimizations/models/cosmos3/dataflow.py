# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations
import torch
from cosmos_framework.utils import log
from cosmos_framework.data.generator.dataflow.loader import (
    _DataflowIterableDataset, SimpleBatcher, DefaultBatchCollator,
)
from cosmos_framework.data.generator.dataflow.base import (
    DataDistributor, RawItemProcessor, SampleBatcher, BatchCollator,
)


def initialize(
    self,
    distributor: DataDistributor,
    processor: RawItemProcessor,
    batcher: SampleBatcher | None = None,
    collator: BatchCollator | None = None,
    batch_size: int | None = None,
    num_workers: int = 0,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    pin_memory: bool = False,
    parallel_dims=None,
    multiprocessing_context: str | None = None,
):
    if batch_size is not None and batcher is not None:
        raise ValueError(
            "Pass either batch_size= (sugar) or an explicit batcher=, not both."
        )
    if batch_size is None and batcher is None:
        raise ValueError("Provide either a batcher= or a batch_size=.")
    if batch_size is not None:
        batcher = SimpleBatcher(batch_size=batch_size)
    if collator is None:
        collator = DefaultBatchCollator()

    if parallel_dims is not None:
        dp_rank, dp_world_size = parallel_dims.dp_coord
    elif torch.distributed.is_initialized():
        dp_rank = torch.distributed.get_rank()
        dp_world_size = torch.distributed.get_world_size()
        if dp_world_size > 1:
            log.info(
                "CosmosDataLoader: using global rank for DP sharding. "
                "For FSDP+TP/PP pass parallel_dims= for the correct DP rank.",
                rank0_only=True,
            )
    else:
        dp_rank, dp_world_size = 0, 1

    dataset = _DataflowIterableDataset(
        distributor=distributor,
        processor=processor,
        batcher=batcher,
        collator=collator,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )

    from cosmos_framework.data.generator.dataflow.distributors import MapDistributor

    if isinstance(distributor, MapDistributor) and num_workers > 0 and not persistent_workers:
        log.info(
            "CosmosDataLoader: MapDistributor requires persistent_workers=True for "
            "correct stateful resume; overriding to True.",
            rank0_only=True,
        )
        persistent_workers = True

    if persistent_workers and num_workers == 0:
        log.info(
            "CosmosDataLoader: persistent_workers=True ignored because num_workers=0.",
            rank0_only=True,
        )
        persistent_workers = False

    loader_kwargs: dict = dict(
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    if num_workers > 0 and multiprocessing_context is not None:
        # "spawn" is required when workers touch a CUDA/HIP context that
        # the parent has already initialized (e.g. torchcodec GPU decoding
        # after torch.distributed init) — fork'd workers inherit an invalid
        # device context and blow up on the first stream sync.
        loader_kwargs["multiprocessing_context"] = multiprocessing_context
    torch.utils.data.DataLoader.__init__(self, dataset, batch_size=None, **loader_kwargs)
