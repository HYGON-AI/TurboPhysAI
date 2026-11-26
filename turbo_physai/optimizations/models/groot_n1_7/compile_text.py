"""Compile text decoder forwards and record per-layer activation."""

import logging
import os
from functools import wraps

from .compile import compile_method_wrapper

logger = logging.getLogger(__name__)


def compile_text_wrapper(original, options):
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    compiled = compile_method_wrapper(original, options)

    @wraps(original)
    def forward(self, *args, **kwargs):
        count = getattr(self, "_turbo_physai_text_compile_calls", 0)
        if count == 0:
            logger.info("PhysAI GR00T text layer %s: compiled forward active", getattr(self, "layer_idx", "unknown"))
        self._turbo_physai_text_compile_calls = count + 1
        return compiled(self, *args, **kwargs)

    return forward
