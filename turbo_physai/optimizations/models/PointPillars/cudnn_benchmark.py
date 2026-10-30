"""Set backend selection only during the wrapped training call."""
from functools import wraps
import torch


def cudnn_benchmark_wrapper(original, options):
    options = options or {}
    benchmark = options.get("benchmark", True)
    deterministic = options.get("deterministic", False)
    if not isinstance(benchmark, bool) or not isinstance(deterministic, bool):
        raise ValueError("backend flags must be boolean")

    @wraps(original)
    def wrapped(*args, **kwargs):
        previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        try:
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = deterministic
            return original(*args, **kwargs)
        finally:
            torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous
    return wrapped
