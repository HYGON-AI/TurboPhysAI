# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
OpenVLA skip-FA2-unpad replacement for TurboPhysAI.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Callable, Optional


def _is_right_padded(attention_mask: Any) -> bool:
    """True when the batch carries no left padding.

    Dropping the mask is only sound for right-padded batches.  With left
    padding the real tokens sit *after* the pad slots, so plain causal attention
    would let them attend to those pad key/values instead of having them
    unpadded away by the varlen path -- silently, and with no error.  A
    right-padded (or unpadded) batch has no zero in its first column.

    The ``.item()`` is a scalar D2H sync, far cheaper than the one this Group
    removes: the varlen path needs a data-dependent ``torch.nonzero`` plus
    ``seqlens_in_batch.max().item()`` on every step.
    """

    return attention_mask.ndim == 2 and bool(attention_mask[:, 0].all().item())


def make_fast_fa2_causal_mask_wrapper(
    original: Callable, options: Optional[Mapping[str, Any]] = None
) -> Callable:
    """Wrapper factory ``(original, options) -> wrapped _update_causal_mask``.

    Enabling the Group installs the returned callable in place of
    ``LlamaModel._update_causal_mask``.  ``options`` (Group options) is accepted
    for framework ``wrap`` compatibility and intentionally unused.
    """

    del options

    @functools.wraps(original)
    def wrapper(self, attention_mask, input_tensor, cache_position, past_seen_tokens):
        # Prefill (no KV cache) + FA2 + an explicit mask: skip the causal-mask
        # materialisation entirely so FA2 never enters the varlen/unpad path.
        # Right padding is a precondition, not an assumption: the mask is
        # dropped, so a left-padded batch must fall back to the original.
        if (
            self.config._attn_implementation == "flash_attention_2"
            and attention_mask is not None
            and past_seen_tokens == 0
            and _is_right_padded(attention_mask)
        ):
            return None
        return original(self, attention_mask, input_tensor, cache_position, past_seen_tokens)

    return wrapper
