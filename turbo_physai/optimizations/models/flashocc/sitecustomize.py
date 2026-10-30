# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Install FlashOCC compatibility in each Python rank started by torchrun."""

import os


if os.environ.get("TURBO_PHYSAI_BOOTSTRAP") == "1":
    import flashocc_optimization
