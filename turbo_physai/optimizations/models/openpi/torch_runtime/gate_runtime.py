# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility entry for the model-level Opt-in exact-expression gate policy; retain native calls for every other case."""

from ..gemma_gate_lookup import dispatch, expression_key
