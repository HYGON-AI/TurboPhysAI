# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Python 3.10 TOML compatibility shared by startup and Cosmos catalogs."""

import importlib
import sys
from types import ModuleType


def ensure_tomllib() -> ModuleType:
    """Preserve an existing tomllib; use tomli only if tomllib is absent.

    Never replace an available stdlib/backport module, and never hide a broken
    module's transitive import error. This also works in spawned data workers.
    """
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError as exc:
        if exc.name != "tomllib":
            raise
    implementation = importlib.import_module("tomli")
    return sys.modules.setdefault("tomllib", implementation)
