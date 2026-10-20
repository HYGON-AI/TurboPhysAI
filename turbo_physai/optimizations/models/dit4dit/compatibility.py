# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Check model files even when a group's target is an adapter or dependency."""

import hashlib
from pathlib import Path

from turbo_physai.engine.contracts import CompatibilityResult

BASELINE_COMMIT = "d8854a11713f846b8b3f768fc3bf2c6cba2bc17f"
# Code hashes, not configuration hashes: customer data/model paths and training
# arguments may change without allowing an already optimized implementation.
BASELINE_FILES = {
    "DiT4DiT/dataloader/__init__.py": "7d6bd3f5c5b4bff980eff5887494af04b72d3ff7d652a4ab9f1eb5f712a5ed41",
    "DiT4DiT/model/modules/vlm/Cosmos25.py": "cdf58e6a195562ecb2802e93080f68166bb8cba9bbc746f4b481b18d94d59876",
    "DiT4DiT/training/train.py": "622410efb8732d3a9da99179c12ab9128d6b4f77f5d865dd4663202015364cb2",
}


def baseline(context):
    import DiT4DiT
    root = Path(DiT4DiT.__file__).resolve().parent.parent
    actual = {}
    for name in BASELINE_FILES:
        path = root / name
        actual[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return CompatibilityResult(
        actual == BASELINE_FILES, expected=BASELINE_FILES, actual=actual,
        reason="DiT4DiT optimization targets must match " + BASELINE_COMMIT,
    )
