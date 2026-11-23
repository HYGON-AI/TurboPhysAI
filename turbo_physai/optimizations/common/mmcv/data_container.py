# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Teach the DataLoader pin-memory thread to pin MMCV ``DataContainer`` data.

``torch.utils.data._utils.pin_memory.pin_memory`` recurses through tensors,
mappings and sequences, and otherwise only calls ``data.pin_memory()`` when the
object provides it.  ``DataContainer`` has no such method, so every batch
produced by ``mmcv.parallel.collate`` (a dict of DataContainers) stays in
pageable memory even with ``pin_memory=True``.

Torch resolves ``pin_memory`` through module globals both in
``_pin_memory_loop`` and in its own recursion, so wrapping that one module
attribute makes nested DataContainers inside dicts/lists pinned as well,
without modifying MMCV.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping


def pin_memory_wrapper(
    original: Callable[..., Any], options: Mapping[str, Any]
) -> Callable[..., Any]:
    """Return a torch ``pin_memory`` that unwraps and rebuilds DataContainers."""

    del options
    from mmcv.parallel.data_container import DataContainer

    @functools.wraps(original)
    def pin_memory(data: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(data, DataContainer):
            return original(data, *args, **kwargs)
        # cpu_only payloads (e.g. img_metas) never reach the device, so
        # pinning them would only spend host memory and copy time.
        if data.cpu_only:
            return data
        return DataContainer(
            original(data.data, *args, **kwargs),
            stack=data.stack,
            padding_value=data.padding_value,
            cpu_only=False,
            pad_dims=data.pad_dims,
        )

    return pin_memory


__all__ = ["pin_memory_wrapper"]
