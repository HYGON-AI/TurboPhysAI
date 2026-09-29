# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RankPartitionedDataLoader worker-kwargs hygiene (adapted source port)."""

from __future__ import annotations

from typing import Any

import torch

import cosmos_framework.data.generator.joint_dataloader as _jdl_mod

from ._source_binding import bind_missing_globals

def rank_partitioned_init(
    self,
    datasets: dict[str, dict[str, Any]],
    **dataloader_kwargs: Any,
):
    """
    Args:
        datasets: Mapping of dataset name to config dict with keys:

            - ``"dataset"`` (required): a lazy config or dataset instance.
            - ``"ratio"`` (required): positive int weight.
            - ``"dataloader_kwargs"`` (optional): dict of keyword arguments
              that override the top-level ``**dataloader_kwargs`` for this
              dataset only (e.g. different ``num_workers`` or ``batch_size``).

        **dataloader_kwargs: Default kwargs forwarded to
            ``torch.utils.data.DataLoader``. ``collate_fn`` defaults to
            ``custom_collate_fn`` if not given.
    """
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    log.info(f"RankPartitionedDataLoader: world_size: {world_size} and rank: {rank}", rank0_only=False)

    _VALID_KEYS = {"dataset", "ratio", "dataloader_kwargs"}
    names: list[str] = []
    dataset_configs: list[Any] = []
    ratios: list[int] = []
    per_dataset_kwargs: list[dict[str, Any]] = []
    for name, cfg in datasets.items():
        extra = set(cfg.keys()) - _VALID_KEYS
        assert not extra, f"Dataset {name!r}: unexpected keys {extra}. Allowed: {_VALID_KEYS}"
        if cfg["ratio"] <= 0:
            log.warning(
                f"RankPartitionedDataLoader: Skipping dataset {name} with ratio {cfg['ratio']}", rank0_only=False
            )
            continue
        names.append(name)
        dataset_configs.append(cfg["dataset"])
        ratios.append(cfg["ratio"])
        per_dataset_kwargs.append(cfg.get("dataloader_kwargs", {}))

    assert len(names) > 0, "No datasets with positive ratios provided."
    assert world_size >= len(names), (
        f"world_size ({world_size}) must be >= number of datasets ({len(names)}) "
        f"so each dataset gets at least one rank."
    )

    total_ratio = sum(ratios)
    ideal = [r / total_ratio * world_size for r in ratios]
    allocations = [max(1, int(q)) for q in ideal]
    remaining = world_size - sum(allocations)
    if remaining > 0:
        remainders = sorted(range(len(ratios)), key=lambda i: ideal[i] - allocations[i], reverse=True)
        for j in range(remaining):
            allocations[remainders[j]] += 1
    elif remaining < 0:
        deficit = -remaining
        while deficit > 0:
            best = max(
                (i for i in range(len(allocations)) if allocations[i] > 1),
                key=lambda i: (allocations[i] - ideal[i], allocations[i]),
            )
            allocations[best] -= 1
            deficit -= 1

    expected_ratios = [r / total_ratio for r in ratios]
    actual_ratios = [a / world_size for a in allocations]
    lines = [f"RankPartitionedDataLoader allocation ({world_size} GPUs):"]
    start = 0
    for i, (name, alloc) in enumerate(zip(names, allocations)):
        end = start + alloc - 1
        lines.append(
            f"  {name} (ratio {ratios[i]}): ranks {start}-{end} ({alloc} GPUs) "
            f"| expected {expected_ratios[i]:.2%}, actual {actual_ratios[i]:.2%}"
        )
        start += alloc
    log.info("\n".join(lines), rank0_only=False)

    cumulative = 0
    my_dataset_idx = -1
    for i, alloc in enumerate(allocations):
        if rank < cumulative + alloc:
            my_dataset_idx = i
            break
        cumulative += alloc
    assert my_dataset_idx >= 0

    shard_rank = rank - cumulative
    shard_world_size = allocations[my_dataset_idx]

    dataset: Any = instantiate(dataset_configs[my_dataset_idx])
    dataset.shard_world_size = shard_world_size
    dataset.shard_rank = shard_rank
    dataset.shard_id = my_dataset_idx

    merged_kwargs = {**dataloader_kwargs, **per_dataset_kwargs[my_dataset_idx]}
    # Worker-only options must be removed after per-dataset overrides.
    # This keeps single-process debugging usable with a spawn-based recipe.
    if merged_kwargs.get("num_workers", 0) == 0:
        merged_kwargs.pop("multiprocessing_context", None)
        merged_kwargs.pop("prefetch_factor", None)
        merged_kwargs["persistent_workers"] = False
    merged_kwargs.setdefault("collate_fn", custom_collate_fn)
    self.dataloader = torch.utils.data.DataLoader(dataset, **merged_kwargs)
    self.dataset_name = names[my_dataset_idx]
    self.dataset = dataset


bind_missing_globals(globals(), _jdl_mod)
