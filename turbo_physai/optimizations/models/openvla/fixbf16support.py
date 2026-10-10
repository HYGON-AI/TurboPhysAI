# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def check_bloat16_supported() -> bool:
    try:
        import packaging.version
        import torch.cuda.nccl as nccl
        import torch.distributed as dist

        if torch.version.cuda:
            return (
                torch.cuda.is_bf16_supported()
                and (packaging.version.parse(torch.version.cuda).release >= (11, 0))
                and dist.is_nccl_available()
                and (nccl.version() >= (2, 10))
            )
        elif torch.version.hip:
            return (
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
                and dist.is_nccl_available()
                and (nccl.version() >= (2, 10))
            )
        else:
            return False

    except Exception:
        return False