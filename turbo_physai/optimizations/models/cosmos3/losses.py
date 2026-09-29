# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""cross_entropy_loss / VLM loss logging with ordered HIP all-reduce."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

import cosmos_framework.model.generator.algorithm.loss.cross_entropy as _ce_mod
import cosmos_framework.model.generator.vlm_model as _vlm_mod

from ._source_binding import bind_missing_globals
from .fsdp_comm import ordered_all_reduce
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_scaling_factor: float = 1.0,
    dp_group: dist.ProcessGroup | None = None,
    cp_group: dist.ProcessGroup | None = None,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Next-token-prediction CE loss with DP/CP group reduction.

    Matches the behavior of cosmos_rl.policy.trainer.llm_trainer.sft_trainer.async_safe_ce
    with the TORCH_CROSS_ENTROPY backend (F.cross_entropy with float32 cast).

    Args:
        logits: (B, T, V) float tensor — raw model output before softmax.
        labels: (B, T) long tensor — ground-truth token ids.
                Positions equal to ignore_index are excluded from the loss.
        loss_scaling_factor: scalar multiplied into the returned loss.
        dp_group: FSDP data-parallel shard group for loss normalization.
                  None = no DP reduction (single-GPU or replicate-only).
        cp_group: Context-parallel group. If size > 1, use per-rank mean.
                  None = no CP reduction.
        ignore_index: label value to exclude (defaults to ``IGNORE_INDEX``, -100).

    Returns:
        Scalar loss tensor.
    """
    # Shift for next-token prediction: predict token[t+1] using hidden state[t].
    # logits[:, :-1] aligns with labels[:, 1:].
    # Reference: async_safe_ce:63-73 (output[:, :-1], target[:, 1:])
    shifted_logits = logits[:, :-1].contiguous().view(-1, logits.size(-1))
    shifted_labels = labels[:, 1:].contiguous().view(-1)

    if cp_group is not None and cp_group.size() > 1:
        # CP path: each rank sees a different sequence segment.
        # Use simple mean reduction; nan_to_num handles fully-ignored batches.
        # Reference: async_safe_ce:74-88
        loss = F.cross_entropy(
            shifted_logits.float(),
            shifted_labels,
            ignore_index=ignore_index,
            reduction="mean",
        )
        loss = torch.nan_to_num(loss, nan=0.0)
        return loss * loss_scaling_factor

    # No-CP path: per-token loss, then normalize over the global valid-token count.
    # Reference: async_safe_ce:89-109
    per_token_loss = F.cross_entropy(
        shifted_logits.float(),
        shifted_labels,
        ignore_index=ignore_index,
        reduction="none",
    )
    n_valid_tokens = (shifted_labels != ignore_index).sum()
    num_dp_workers = 1
    if dp_group is not None:
        if torch.version.hip is not None:
            ordered_all_reduce(n_valid_tokens, op=dist.ReduceOp.SUM, group=dp_group)
        else:
            dist.all_reduce(n_valid_tokens, op=dist.ReduceOp.SUM, group=dp_group)
        num_dp_workers = dist.get_world_size(group=dp_group)

    loss = per_token_loss.sum() / (n_valid_tokens + 1e-8) * (num_dp_workers * loss_scaling_factor)
    return loss


def vlm_training_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
    """position_ids → forward → CE loss."""
    position_ids = get_position_ids(
        self.hf_config,
        input_ids=data["input_ids"],
        image_grid_thw=data.get("image_grid_thw"),
        video_grid_thw=data.get("video_grid_thw"),
        attention_mask=data.get("attention_mask"),
    )
    if position_ids is not None:
        data["position_ids"] = position_ids

    labels = data.pop("labels")
    data.pop("attention_mask", None)
    logits = self.model(**data)
    loss = self._loss_fn(logits, labels)

    # loss_avg: DP-averaged loss for logging (matches cosmos-rl ReduceOp.AVG).
    # Does not affect the backward scalar. Pick the same 1-D sub-mesh the
    # legacy single-mesh ``ParallelDims.dp_mesh`` returned — dp_shard if
    # sharding is on, else dp_replicate — so the reduction group is
    # byte-identical to pre-merge behavior.
    loss_avg = loss.detach().clone()
    pd = getattr(self, "parallel_dims", None)
    dp_mesh = pd.dp_mesh if pd is not None else None
    if torch.distributed.is_initialized() and dp_mesh is not None:
        sub_dim = "dp_shard" if pd.dp_shard_enabled else "dp_replicate"
        loss_group = dp_mesh[sub_dim].get_group()
        if torch.version.hip is not None:
            ordered_all_reduce(loss_avg, op=torch.distributed.ReduceOp.AVG, group=loss_group)
        else:
            torch.distributed.all_reduce(loss_avg, op=torch.distributed.ReduceOp.AVG, group=loss_group)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        log.info(f"train/loss_avg: {loss_avg.item():.5f} (iteration {iteration})")

    return {"loss": loss, "loss_avg": loss_avg, "labels": labels}, loss


bind_missing_globals(globals(), _ce_mod, _vlm_mod)
