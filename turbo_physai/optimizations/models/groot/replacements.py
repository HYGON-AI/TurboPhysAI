"""GR00T runtime compatibility implementations."""

import os
from functools import wraps


def dit_sdpa_context_wrapper(original, group_options):
    """Use math SDPA for DiT on HIP, while preserving the model override."""
    @wraps(original)
    def sdpa_context():
        import torch

        if torch.version.hip is not None and os.environ.get("GR00T_DIT_SDPA_MODE") != "default":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.MATH)
        return original()

    return sdpa_context
