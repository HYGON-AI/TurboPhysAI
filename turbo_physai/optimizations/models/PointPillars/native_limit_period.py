"""Tensor-native limit_period wrapper used by compiled PointPillars paths."""

from functools import wraps

import numpy as np
import torch


def native_limit_period_wrapper(original, options):
    """Bypass array_converter for tensors while preserving NumPy behavior."""
    @wraps(original)
    def wrapped(val, offset=0.5, period=np.pi):
        if isinstance(val, torch.Tensor):
            return val - torch.floor(val / period + offset) * period
        if isinstance(val, np.ndarray):
            return val - np.floor(val / period + offset) * period
        return original(val, offset, period)

    return wrapped
