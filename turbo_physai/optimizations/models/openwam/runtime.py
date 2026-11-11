# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

"""Validated numeric options supplied by Turbo Groups."""

from .settings import value


def integer(name, default, minimum=0):
    raw = value(name, default)
    if raw is None:
        raw = default
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise ValueError(
            f"OpenWAM option {name} must be an integer >= {minimum}; got {raw!r}"
        )
    return raw
