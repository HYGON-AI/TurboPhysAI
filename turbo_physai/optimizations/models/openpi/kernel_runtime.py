# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Compose audited Inductor dispatches without patching global targets."""

from functools import wraps

from . import gelu_lookup, rope_trig_cache, softmax_backward_reuse, softmax_forward_schedule
from .torch_runtime.installation import _prepare_runtime


def supports_kernel_runtime(self, *args, stream, benchmark_run=False, **kwargs):
    """Enable the audited kernel replacements only on their target device."""
    del args, stream, benchmark_run, kwargs
    props = getattr(self, "device_props", None)
    return bool(
        getattr(props, "cc", None) == "gfx938"
        and getattr(props, "multi_processor_count", None) == 64
    )


def wrap_kernel_runtime(original, options):
    del options
    dispatch = None

    @wraps(original)
    def wrapped(tuner, *args, stream, **kwargs):
        nonlocal dispatch
        if dispatch is None:
            _prepare_runtime()
            dispatch = gelu_lookup.wrap_gelu_lookup(original, {})
            dispatch = rope_trig_cache.make_run(dispatch)
            dispatch = softmax_forward_schedule.make_run(dispatch)
            dispatch = softmax_backward_reuse.make_run(dispatch)
        return dispatch(tuner, *args, stream=stream, **kwargs)

    return wrapped
