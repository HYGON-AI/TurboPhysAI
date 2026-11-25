# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime capability probes for the MapTRv2 optimization Groups.

``runtime_condition`` targets declared in :mod:`maptrv2.catalog`
are called with the arguments of the wrapped call and must return ``bool``.
A condition that returns ``False`` selects the unmodified upstream callable.

Dependencies are imported lazily. Catalog discovery does not load Torch;
selected compatibility checks and runtime probes may load it.
"""

from __future__ import annotations

import os


def torch_compile_available(*_args, **_kwargs) -> bool:
    """Return ``True`` when ``torch.compile`` may be used in this process."""

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return False
    try:
        import torch
    except Exception:
        return False
    return callable(getattr(torch, "compile", None))


def dynamo_available(*_args, **_kwargs) -> bool:
    """Return ``True`` when ``torch._dynamo.disable`` is available."""

    try:
        import torch
    except Exception:
        return False
    dynamo = getattr(torch, "_dynamo", None)
    return callable(getattr(dynamo, "disable", None))


def static_assigner_available(*_args, **_kwargs) -> bool:
    """Return whether the complete static assigner path can run."""

    if os.getenv("TURBO_PHYSAI_DISABLE_ASSIGNER_STATIC", "0") == "1":
        return False
    return torch_compile_available() and dynamo_available()


def pv_mask_sampling_enabled(*_args, **_kwargs) -> bool:
    """Return ``True`` when the vectorized PV-mask resampling may be used."""

    return os.getenv("TURBO_PHYSAI_DISABLE_PV_MASK", "0") != "1"


def shapely_query_returns_indices(*_args, **_kwargs) -> bool:
    """Return ``True`` when ``STRtree.query`` yields indices (Shapely >= 2)."""

    if os.getenv("TURBO_PHYSAI_DISABLE_EVAL_COMPAT", "0") == "1":
        return False
    try:
        import shapely
    except Exception:
        return False
    try:
        major = int(str(shapely.__version__).split(".", 1)[0])
    except (AttributeError, ValueError):
        return False
    return major >= 2


__all__ = [
    "dynamo_available",
    "pv_mask_sampling_enabled",
    "shapely_query_returns_indices",
    "static_assigner_available",
    "torch_compile_available",
]
