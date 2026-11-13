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
LeRobotDataset._query_hf_dataset：只解码需要的那一列，避免图像列一起解码。
"""

from __future__ import annotations


def query_hf_dataset(self, query_indices):
    import torch
    from lingbotvla.data.vla_data.base_dataset import _to_relative_indices

    result: dict = {}
    for key, q_idx in query_indices.items():
        if key in self.meta.video_keys:
            continue
        relative_indices = _to_relative_indices(self, q_idx)
        if key in self.hf_dataset.column_names:
            sub = self.hf_dataset.select_columns([key])
        else:
            sub = self.hf_dataset
        result[key] = torch.stack(sub[relative_indices][key])
    return result

class _ColumnView:
    __slots__ = ("_ds", "_cols", "_sub")

    def __init__(self, ds, cols):
        self._ds = ds
        self._cols = cols
        self._sub = None

    def _view(self):
        if self._sub is None:
            self._sub = self._ds.select_columns(self._cols)
        return self._sub

    def __getitem__(self, key):
        if isinstance(key, int) and not isinstance(key, bool):
            return self._view()[key]
        return self._ds[key]

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, name):
        return getattr(self._ds, name)


def _install_row_pruning(dataset) -> None:
    delta = getattr(dataset, "delta_indices", None)
    if not delta:
        return
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None or isinstance(hf, _ColumnView):
        return
    try:
        cameras = list(dataset.meta.camera_keys)
        columns = list(hf.column_names)
    except Exception:
        return
    if not cameras or not all(cam in delta and cam in columns for cam in cameras):
        return
    keep = [c for c in columns if c not in cameras]
    if len(keep) == len(columns):
        return
    dataset.hf_dataset = _ColumnView(hf, keep)


def wrap_dataset_init(original, _options=None):
    import functools

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        try:
            _install_row_pruning(self)
        except Exception as exc:
            print(f"[fast-hf-query] row pruning skipped: {exc!r}", flush=True)

    return __init__


__all__ = ["query_hf_dataset", "wrap_dataset_init"]
