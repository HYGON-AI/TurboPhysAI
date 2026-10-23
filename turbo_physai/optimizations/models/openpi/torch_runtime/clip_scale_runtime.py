# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility entry for the model-level Opt-in clipping scaling: leave native norms, coefficients and grouping intact."""

from ..clip_grad_scale import _install_runtime as install
