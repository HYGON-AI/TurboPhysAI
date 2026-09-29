# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy import path for the shared, idempotent TOML fallback.

This module must not be installed with ``replace_import``: tomllib can already
be loaded by interpreter startup or another dependency.
"""

from ....bootstrap._toml import ensure_tomllib

_impl = ensure_tomllib()

load = _impl.load
loads = _impl.loads
TOMLDecodeError = _impl.TOMLDecodeError

__all__ = ["TOMLDecodeError", "load", "loads"]
