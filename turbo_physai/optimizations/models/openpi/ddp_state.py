"""Shared dispatcher for the two independent DDP optimizations.

Turbo applies wrappers targeting the same ``DistributedDataParallel.__init__``
from the original target snapshot, so the later wrapper can otherwise replace
the earlier one. Keep the group switches independent while installing one
dispatcher that composes whichever groups are enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import Any


_ORIGINAL: Any | None = None
_STATIC_OPTIONS: Mapping[str, Any] | None = None
_COMPILE_OPTIONS: Mapping[str, Any] | None = None


def _register_original(original: Any) -> None:
    global _ORIGINAL
    if _ORIGINAL is None:
        _ORIGINAL = original


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def configured_ddp_kwargs(base: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the active static-DDP options over a caller's upstream kwargs.

    The deferred-metric replacement constructs DDP itself, so relying only on
    a constructor wrapper would leave that replacement's hard-coded kwargs in
    effect.  Keeping this helper in the shared dispatcher preserves the group
    switch: with the static group disabled, the caller's original kwargs are
    returned unchanged.
    """

    call_kwargs = dict(base or {})
    if _STATIC_OPTIONS is not None:
        call_kwargs.update(
            broadcast_buffers=_bool(
                _STATIC_OPTIONS.get("broadcast_buffers"), False
            ),
            find_unused_parameters=_bool(
                _STATIC_OPTIONS.get("find_unused_parameters"), False
            ),
            gradient_as_bucket_view=_bool(
                _STATIC_OPTIONS.get("gradient_as_bucket_view"), True
            ),
            static_graph=_bool(_STATIC_OPTIONS.get("static_graph"), True),
        )
    return call_kwargs


def configure_optimize_ddp() -> None:
    """Apply the configured torch.compile DDP mode before DDP construction."""

    if _COMPILE_OPTIONS is None:
        return
    import torch

    mode = _COMPILE_OPTIONS.get("optimize_ddp", "no_optimization")
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in {"0", "false", "off"}:
            mode = False
        elif normalized == "no_optimization":
            mode = "no_optimization"
        elif normalized in {"1", "true", "on", "default"}:
            mode = True
    torch._dynamo.config.optimize_ddp = mode


def register_compile_options(options: Mapping[str, Any]) -> None:
    """Register compile-mode options without wrapping DDP.__init__."""

    global _COMPILE_OPTIONS
    _COMPILE_OPTIONS = options


def _dispatcher(fallback: Any):
    _register_original(fallback)

    @wraps(fallback)
    def wrapped(self, *args, **kwargs):
        configure_optimize_ddp()
        call_kwargs = configured_ddp_kwargs(kwargs)

        original = _ORIGINAL if _ORIGINAL is not None else fallback
        return original(self, *args, **call_kwargs)

    return wrapped


def register_static(original: Any, options: Mapping[str, Any]):
    global _STATIC_OPTIONS
    _register_original(original)
    _STATIC_OPTIONS = options
    return _dispatcher(original)


def register_compile(original: Any, options: Mapping[str, Any]):
    global _COMPILE_OPTIONS
    _register_original(original)
    _COMPILE_OPTIONS = options
    return _dispatcher(original)
