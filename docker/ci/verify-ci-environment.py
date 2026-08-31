#!/usr/bin/env python3
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime gate for TurboPhysAI HCU CI after the candidate is installed."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-turbophysai",
        action="store_true",
        help="also require the CI job's just-built TurboPhysAI package and ops",
    )
    args = parser.parse_args()

    import hipdnn  # noqa: F401
    import lightop  # noqa: F401
    import torch

    print(f"PyTorch: {torch.__version__}")
    print(f"HCU available: {torch.cuda.is_available()}")
    print(f"HCU count: {torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("an HCU device is required for this CI runtime gate")

    if args.require_turbophysai:
        import turbo_physai
        import turbo_physai.ops

        print(f"TurboPhysAI: {turbo_physai.__file__}")
        print(f"TurboPhysAI ops: {turbo_physai.ops.__file__}")
    print("TurboPhysAI CI runtime gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
