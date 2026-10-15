# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Optional

from torch.distributed.fsdp import BackwardPrefetch

logger = logging.getLogger("openvla.fsdp.prefetch")

_BACKWARD_PREFETCH = {
    "pre": BackwardPrefetch.BACKWARD_PRE,
    "post": BackwardPrefetch.BACKWARD_POST,
    "none": None,
}


def _prefetch_options(options: Optional[Mapping[str, Any]]) -> tuple[bool, bool, str]:
    """Parse the Group ``options`` into ``(limit_all_gathers, forward_prefetch, backward_prefetch)``.

    Defaults are the overlap-optimized values: the Group being enabled *is* the
    opt-in, so an empty ``options`` mapping means "apply the optimization".
    """
    options = dict(options or {})
    limit_all_gathers = bool(options.get("limit_all_gathers", False))
    forward_prefetch = bool(options.get("forward_prefetch", True))

    raw_backward_prefetch = options.get("backward_prefetch", "pre")
    if raw_backward_prefetch is None:
        raw_backward_prefetch = "none"
    backward_prefetch = str(raw_backward_prefetch).lower()
    if backward_prefetch not in _BACKWARD_PREFETCH:
        raise ValueError(
            "openvla.fsdp.prefetch: options.backward_prefetch must be one of "
            f"{sorted(_BACKWARD_PREFETCH)}, got {raw_backward_prefetch!r}"
        )
    return limit_all_gathers, forward_prefetch, backward_prefetch


def fsdp_prefetch_wrapper(original: Any, options: Optional[Mapping[str, Any]] = None):
    """Wrapper factory ``(original, options) -> FSDP class with overlapped collectives``.

    ``original`` is ``torch.distributed.fsdp.FullyShardedDataParallel``.  The
    returned class injects the three communication keyword defaults into every
    construction while leaving the submitted values untouched.
    """
    if not isinstance(original, type):
        return original

    limit_all_gathers, forward_prefetch, backward_prefetch = _prefetch_options(options)
    backward_prefetch_value = _BACKWARD_PREFETCH[backward_prefetch]

    def prefetch_init(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("limit_all_gathers", limit_all_gathers)
        kwargs.setdefault("forward_prefetch", forward_prefetch)
        kwargs.setdefault("backward_prefetch", backward_prefetch_value)
        original.__init__(self, *args, **kwargs)

    # Build the subclass through `type()` so the installed class keeps the exact
    # original name (torch/dynamo internals and log lines that key off
    # `type(module).__name__` stay indistinguishable from the unpatched run).
    replacement = type(
        original.__name__,
        (original,),
        {
            "__init__": prefetch_init,
            "__doc__": (
                f"{original.__doc__ or ''}\n\n"
                "[openvla.fsdp.prefetch] Injected FSDP1 communication defaults: "
                f"limit_all_gathers={limit_all_gathers}, "
                f"forward_prefetch={forward_prefetch}, "
                f"backward_prefetch={backward_prefetch!r}."
            ),
            "__module__": original.__module__,
        },
    )

    logger.info(
        "FSDP1 wrapped with `openvla.fsdp.prefetch` (limit_all_gathers=%s, "
        "forward_prefetch=%s, backward_prefetch=%s)",
        limit_all_gathers,
        forward_prefetch,
        backward_prefetch,
    )
    return replacement
