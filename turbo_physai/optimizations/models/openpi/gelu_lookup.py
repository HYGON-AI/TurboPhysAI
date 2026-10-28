# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compose the Gemma gate and SigLIP GELU lookup dispatchers."""

from functools import wraps

from . import gemma_gate_lookup, siglip_gelu_lookup
from .torch_runtime.installation import _prepare_runtime


def wrap_gelu_lookup(original, options):
    del options
    dispatch = None

    def gate_dispatch(tuner, *args, stream, **kwargs):
        name = tuner.inductor_meta.get("kernel_name", "")
        if "gelu" in name and any(
            isinstance(value, int) and value == 507510784 for value in args
        ):
            return gemma_gate_lookup.dispatch(tuner, original, args, stream, kwargs)
        return original(tuner, *args, stream=stream, **kwargs)

    @wraps(original)
    def wrapped(tuner, *args, stream, **kwargs):
        nonlocal dispatch
        if dispatch is None:
            _prepare_runtime()
            dispatch = siglip_gelu_lookup.wrap_run(gate_dispatch)
        return dispatch(tuner, *args, stream=stream, **kwargs)

    return wrapped
