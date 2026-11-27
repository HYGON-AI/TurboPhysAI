# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


def _query_with_fallback(hf_dataset, relative, key):
    import torch

    try:
        return torch.stack(hf_dataset.select(relative)[key])
    except (KeyError, TypeError, IndexError):
        return torch.stack(hf_dataset[relative][key])


def query_hf_dataset(self, query_indices):
    import torch

    index_map = getattr(self, "_absolute_to_relative_idx", None)
    result: dict = {}
    for key, q_idx in query_indices.items():
        if key in self.meta.video_keys:
            continue
        relative = q_idx if index_map is None else [index_map[i] for i in q_idx]
        if key in self.hf_dataset.column_names:
            try:
                sub = self.hf_dataset.select_columns([key])
                result[key] = torch.stack(sub[relative][key])
                continue
            except (KeyError, TypeError, IndexError):
                pass
        result[key] = _query_with_fallback(self.hf_dataset, relative, key)
    return result
