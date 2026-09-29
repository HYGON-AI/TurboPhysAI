# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Sequence-packing FlashAttention max_seqlen bucketing (adapted source port)."""

from __future__ import annotations

import torch

import cosmos_framework.data.generator.sequence_packing.runtime as _runtime

from ._source_binding import bind_missing_globals

FLASH_ATTN_MAX_SEQLEN_BUCKETS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)


def _bucket_flash_attn_max_seqlen(n: int) -> int:
    """Bucket Python max_seqlen values passed to FlashAttention.

    ``torch.compile`` guards Python ints by exact value. Keeping the exact
    per-batch max sequence length in compiled decoder layers can therefore
    trigger one graph per batch. The exact lengths remain available through
    ``max_*_len``; only FlashAttention's Python ``max_seqlen`` launch
    arguments use these bucketed values.
    """
    if n < 0:
        raise ValueError(f"FlashAttention max_seqlen cannot be negative, got {n}")
    if n == 0:
        return 0
    for bucket in FLASH_ATTN_MAX_SEQLEN_BUCKETS:
        if n <= bucket:
            return bucket
    raise ValueError(
        f"sequence length {n} exceeds FlashAttention max_seqlen buckets "
        f"{FLASH_ATTN_MAX_SEQLEN_BUCKETS}"
    )


def init_sequence_pack(
    sample_lens: List[int],
    split_lens: List[int],
    attn_modes: List[str],
    device: torch.device,
) -> dict[str, Any]:
    _max_sample_len = max(sample_lens)
    _max_causal_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "causal"), default=0)
    _max_full_len = max((split_lens[i] for i in range(len(split_lens)) if attn_modes[i] == "full"), default=0)

    sample_lens_cu = torch.tensor([0] + sample_lens, device=device, dtype=torch.int32)  # [N_samples+1]
    _sample_offsets = torch.cumsum(sample_lens_cu, dim=0, dtype=torch.int32)  # [N_samples+1]

    _causal_indices, _causal_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "causal", device)
    _full_indices, _full_only_seq_offsets = _compute_mode_indices_and_offsets(split_lens, attn_modes, "full", device)

    return dict(
        sample_offsets=_sample_offsets,
        max_sample_len=_max_sample_len,
        max_causal_len=_max_causal_len,
        max_full_len=_max_full_len,
        fa_max_sample_len=_bucket_flash_attn_max_seqlen(_max_sample_len),
        fa_max_causal_len=_bucket_flash_attn_max_seqlen(_max_causal_len),
        fa_max_full_len=_bucket_flash_attn_max_seqlen(_max_full_len),
        _causal_indices=_causal_indices,
        _full_indices=_full_indices,
        _causal_seq_offsets=_causal_seq_offsets,
        _full_only_seq_offsets=_full_only_seq_offsets,
        _num_causal_tokens=len(_causal_indices),
        _num_full_tokens=len(_full_indices),
        split_lens=split_lens,
        attn_modes=attn_modes,
    )


bind_missing_globals(globals(), _runtime)
