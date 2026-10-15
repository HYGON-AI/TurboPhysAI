# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
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


def checkpoint_skip_layers():
    return integer("checkpoint_skip_layers", 0)


def checkpoint_layers(total):
    skip = checkpoint_skip_layers()
    if skip > total:
        raise ValueError(f"checkpoint_skip_layers={skip} exceeds model layers={total}")
    return total - skip
