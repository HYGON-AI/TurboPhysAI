# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility entry for the model-level Opt-in batched scheduling; tensor arithmetic and BF16 rounding are unchanged."""

from ..batched_adamw import _install_runtime as install
