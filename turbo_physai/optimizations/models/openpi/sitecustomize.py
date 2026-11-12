# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Install OpenPI compatibility shims during Python startup."""

from __future__ import annotations

import datetime as _datetime


def apply_compatibility_shims() -> None:
    """Apply only compatibility fixes needed by this Python/data runtime."""
    # Python 3.10 does not expose datetime.UTC (added in Python 3.11).
    if not hasattr(_datetime, "UTC"):
        _datetime.UTC = _datetime.timezone.utc

    # Older LeRobot parquet metadata can contain _type="List". Newer datasets
    # releases represent the same feature with Sequence but omit the old alias.
    try:
        from datasets.features import features as _features

        feature_types = getattr(_features, "_FEATURE_TYPES", None)
        sequence_type = getattr(_features, "Sequence", None)
        if (
            isinstance(feature_types, dict)
            and "List" not in feature_types
            and sequence_type is not None
        ):
            feature_types["List"] = sequence_type
    except Exception:
        # Non-training interpreters may not have datasets installed.
        pass

apply_compatibility_shims()
