# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    weight = self.proj.weight.view(self.embed_dim, -1)
    hidden_states = hidden_states.reshape(-1, weight.shape[1])
    return F.linear(hidden_states.to(weight.dtype), weight)
