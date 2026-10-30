"""Scope the Dynamo DDP setting to an epoch-based training loop."""
from functools import wraps
import torch._dynamo.config


def disable_dynamo_ddp_wrapper(original, options):
    @wraps(original)
    def wrapped(*args, **kwargs):
        with torch._dynamo.config.patch(optimize_ddp=False):
            return original(*args, **kwargs)
    return wrapped
