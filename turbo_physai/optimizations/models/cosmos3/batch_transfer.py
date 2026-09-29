# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Keep Generator ``image_size`` metadata on CPU during batch transfer (HIP).

The adapted source adds ``misc.training_batch_to_device`` and routes the two
trainer call sites (``ImaginaireTrainer.train`` / ``validate``) through it.
Rewriting the trainer methods would copy the whole loop, so the shared
``misc.to`` is wrapped instead; the reroute triggers only for the exact call
shape the trainer uses (a mapping batch moved by ``device=`` only), leaving
every other ``misc.to`` caller (checkpoint CPU staging, dtype casts) on the
original path.
"""

from __future__ import annotations

import collections.abc
from functools import wraps
from typing import Any

import torch


def wrap_trainer_batch_transfer(original, options):
    """Wrapper for ``cosmos_framework.utils.misc.to``."""
    del options

    @wraps(original)
    def wrapped(data: Any, device=None, dtype=None, memory_format=None, **kwargs) -> Any:
        keep_metadata = (
            bool(getattr(torch.version, "hip", None))
            and device is not None
            and dtype is None
            and memory_format is None
            and not kwargs
            and isinstance(data, collections.abc.Mapping)
            and "image_size" in data
        )
        if not keep_metadata:
            passthrough = {}
            if dtype is not None:
                passthrough["dtype"] = dtype
            if memory_format is not None:
                passthrough["memory_format"] = memory_format
            passthrough.update(kwargs)
            if device is not None:
                passthrough["device"] = device
            return original(data, **passthrough)
        return type(data)(
            {
                key: original(value, device="cpu" if key == "image_size" else device)
                for key, value in data.items()
            }
        )

    return wrapped
