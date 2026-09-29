# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1


"""DCP loader that skips checkpoint writes for disposable performance runs."""

from cosmos_framework.checkpoint.dcp import DistributedCheckpointer


class LoadOnlyDistributedCheckpointer(DistributedCheckpointer):
    """Load through DCP normally, but do not persist training checkpoints."""

    def save(self, model, optimizer, scheduler, grad_scaler, iteration: int) -> None:
        return None
