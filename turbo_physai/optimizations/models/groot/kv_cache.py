"""Avoid creating an implicit KV cache for GR00T text prefill."""

import inspect
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def disable_kv_cache_wrapper(original, options):
    signature = inspect.signature(original)

    @wraps(original)
    def forward(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        if "use_cache" not in bound.arguments and bound.arguments.get("past_key_values") is None:
            kwargs["use_cache"] = False
            count = getattr(self, "_turbo_physai_kv_cache_disabled_calls", 0)
            if count == 0:
                logger.info("PhysAI GR00T: disabled implicit text KV cache")
            self._turbo_physai_kv_cache_disabled_calls = count + 1
        return original(self, *args, **kwargs)

    return forward
