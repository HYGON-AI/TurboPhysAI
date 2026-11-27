# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""把 depth_align_embs 从 CPU 搬到 GPU

官方源码：

    self.depth_align_embs = nn.Parameter(torch.randn(...)).to(dtype=torch.bfloat16)

Parameter.to() 返回的是 Tensor，赋值后走 Module.__setattr__ 的普通tensor 分支，不再是可学习的参数。
既不在 optimizer 的更新列表里，也不被 FSDP 分片，—— 官方源码这里的每步前向只是拿它的值当常量。
这里跟官方语义保持一致，只是在init阶段就搬运至GPU，不在forward中每step进行拷贝。
"""

from __future__ import annotations

import functools
import logging

import torch

logger = logging.getLogger(__name__)


def wrap_init_depth_heads(original, options=None):
    @functools.wraps(original)
    def wrapped(self, config):
        original(self, config)
        param = getattr(self, "depth_align_embs", None)
        if param is None or param.device.type != "cpu" or not torch.cuda.is_available():
            return
        device = torch.device("cuda", torch.cuda.current_device())
        self.depth_align_embs = param.detach().to(device=device)
        logger.info("lingbot-vla: moved depth_align_embs to %s", device)

    return wrapped
