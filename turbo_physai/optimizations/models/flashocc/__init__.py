# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""TurboPhysAI built-in optimization package for FlashOCC."""

from .compatibility_bootstrap import install_for_turbo_run

install_for_turbo_run()

from . import catalog

__all__ = ["catalog"]
