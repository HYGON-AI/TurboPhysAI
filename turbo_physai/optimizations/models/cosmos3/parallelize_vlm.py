# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLM FSDP2 wrapping with fsdp_layers_per_group grouping and ordered HIP
collectives (adapted source port of ``parallelize_vlm.apply_fsdp``)."""

from __future__ import annotations

from itertools import groupby

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

import cosmos_framework.model.generator.parallelize_vlm as _parallelize_vlm

from ._source_binding import bind_missing_globals
from .fsdp_comm import OrderedAllGather, OrderedReduceScatter

def apply_fsdp(
    model: HFModel,
    parallel_dims: ParallelDims,
    parallelism_config: ParallelismConfig,
    precision: str,
) -> None:
    """Apply FSDP2 to an HFModel in-place.

    Uses torch.distributed.fsdp.fully_shard (FSDP2).  Each transformer block is
    sharded individually for fine-grained memory savings; the outer model is then
    wrapped to cover remaining parameters (embeddings, layer norms, lm_head).

    When ``parallelism_config.fsdp_layers_per_group > 1``, consecutive blocks
    are grouped into a single FSDP unit — see the group-shard block below for
    details.

    Supported architectures:
    - Language models: ``inner.model.layers`` (standard HF LLM structure)
    - Vision-language models: additionally ``inner.visual.blocks`` (Qwen3-VL)

    No-op when there is no shard axis (``dp_shard <= 1``): single-GPU, or
    replicate-only (``dp_replicate > 1, dp_shard == 1``) which uses DDP outside
    this function.

    Args:
        model:              HFModel instance (``model`` attribute must be on meta or CPU device).
        parallel_dims:      ParallelDims with meshes already built via
                            :meth:`ParallelDims.build_meshes`.
        parallelism_config: Source of FSDP master dtype (``fsdp_master_dtype``;
                            threaded to ``MixedPrecisionPolicy.reduce_dtype``)
                            and layer group size (``fsdp_layers_per_group``).
        precision:          FSDP MixedPrecisionPolicy parameter dtype
                            (``"bfloat16"``, ``"float16"``, or ``"float32"``).
    """
    if not parallel_dims.dp_shard_enabled:
        log.info("parallelize: dp_shard <= 1 — skipping FSDP2 wrapping")
        return

    mp_policy = MixedPrecisionPolicy(
        param_dtype=PRECISION_TO_TORCH_DTYPE[precision],
        reduce_dtype=PRECISION_TO_TORCH_DTYPE[parallelism_config.fsdp_master_dtype],
    )

    # 2-D (dp_replicate × dp_shard) mesh for HSDP, or 1-D dp_shard sub-mesh
    # for pure FSDP. In the overlay design cp does NOT fold into the FSDP
    # shard axis; cp/cfgp are handled by separate meshes.
    if parallel_dims.dp_replicate_enabled:
        fsdp_mesh = parallel_dims.dp_mesh
    else:
        fsdp_mesh = parallel_dims.dp_shard_mesh
    fsdp_kwargs = {"mesh": fsdp_mesh, "mp_policy": mp_policy}
    hcu_ordered_comm = torch.version.hip is not None
    if hcu_ordered_comm:
        all_gather_comm = OrderedAllGather()
        reduce_scatter_comm = OrderedReduceScatter()

    def fully_shard_with_ordered_comm(module_or_group: "nn.Module | list[nn.Module]") -> None:
        """Wrap a single module OR a list of modules as one FSDP unit.

        For a ``list[nn.Module]`` input, FSDP2 (torch >= 2.4) binds every module
        in the list to ONE shared ``FSDPParamGroup`` — a single all-gather /
        reduce-scatter covers the whole group's params, while each module still
        gets its own forward hook (so the gather fires correctly whenever the
        parent iterates through the group's members). ``set_custom_all_gather``
        writes to ``fsdp_param_group._all_gather_comm``, so calling it on any
        one representative from the group applies to the whole group.
        """
        fully_shard(module_or_group, **fsdp_kwargs)
        if hcu_ordered_comm:
            rep = module_or_group[0] if isinstance(module_or_group, list) else module_or_group
            assert isinstance(rep, FSDPModule)
            rep.set_custom_all_gather(all_gather_comm)
            rep.set_custom_reduce_scatter(reduce_scatter_comm)


    inner = model.model

    # Collect the repeated blocks by their ORIGINAL type name BEFORE any
    # fully_shard call (see _collect_repeated_blocks). apply_compile (if it ran
    # first) used in-place compile, which preserves the type names, so this
    # re-collection still matches.
    blocks, _ = _collect_repeated_blocks(inner)

    # Shard each collected block (reversed = leaf-first), then the root. When
    # fsdp_layers_per_group == 1 (the default) this preserves the historical
    # one-unit-per-block behaviour. For values > 1, consecutive blocks of the
    # SAME type are chunked into groups; each chunk becomes one FSDP unit
    # (one all-gather covers the whole chunk). Chunking is confined to
    # same-type runs so text decoder layers and vision blocks — which execute
    # in different regions of the forward pass — never land in the same group
    # (would stretch the gathered-param lifetime across unrelated regions and
    # waste memory). ``itertools.groupby`` over the depth-first traversal of
    # ``_no_split_modules`` types produces contiguous per-type runs naturally.
    group_size = max(1, int(getattr(parallelism_config, "fsdp_layers_per_group", 1)))
    chunks: list[list[nn.Module]] = []
    for _tname, run_iter in groupby(blocks, key=lambda b: type(b).__name__):
        run = list(run_iter)
        for i in range(0, len(run), group_size):
            chunks.append(run[i : i + group_size])

    # Reversed = leaf-first (last chunk in forward = first in backward).
    for chunk in reversed(chunks):
        if len(chunk) == 1:
            # Single-module path: matches the pre-grouping call shape exactly.
            fully_shard_with_ordered_comm(chunk[0])
        else:
            fully_shard_with_ordered_comm(chunk)
    log.info(
        f"Wrapped {len(blocks)} sub-modules into {len(chunks)} FSDP unit(s) "
        f"(fsdp_layers_per_group={group_size})."
    )

    # Wrap the full inner model to cover remaining parameters
    # (embed_tokens, final layer norm, lm_head, visual projector stem, etc.)
    # NOTE: FSDP-2 CPU offload (offload_policy=CPUOffloadPolicy()) was never
    # wired through to any active recipe and the path was untested; see the
    # comment in vlm_model._init_vlm meta-materialize block (search for
    # "FSDP-2 CPU offload") for how to re-enable it.
    fully_shard_with_ordered_comm(inner)
    log.info("parallelize: FSDP2 applied with one collective submission stream per process group")


bind_missing_globals(globals(), _parallelize_vlm)
