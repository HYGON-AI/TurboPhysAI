# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""External model package; use the installed Turbo Engine without a source overlay."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]

from .compatibility import install_for_turbo_run

install_for_turbo_run()
