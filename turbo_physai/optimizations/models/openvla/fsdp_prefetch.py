# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA FSDP1 通信重叠。

关闭 all_gather 限流（``limit_all_gathers=False``），开启 ``forward_prefetch``
与 ``backward_prefetch=BACKWARD_PRE``，让 all_gather 跑在计算前面，提高通信与
计算的重叠度。这三项是本组固定开启的行为，不读取 Group options。
"""

from __future__ import annotations

from torch.distributed.fsdp import BackwardPrefetch

LIMIT_ALL_GATHERS = False
FORWARD_PREFETCH = True
BACKWARD_PREFETCH = BackwardPrefetch.BACKWARD_PRE


def fsdp_prefetch_wrapper(original, options=None):
    """``wrap`` 工厂：给每次 FSDP 构造注入固定的通信参数。"""

    del options

    if not isinstance(original, type):
        raise TypeError(f"expected the FSDP class, got {original!r}")

    def __init__(self, *args, **kwargs):
        # setdefault：显式传入的调用点优先，只补默认值。
        kwargs.setdefault("limit_all_gathers", LIMIT_ALL_GATHERS)
        kwargs.setdefault("forward_prefetch", FORWARD_PREFETCH)
        kwargs.setdefault("backward_prefetch", BACKWARD_PREFETCH)
        original.__init__(self, *args, **kwargs)

    # 用 type() 建子类，保持类名与 __module__ 不变（日志和 Dynamo 内部按名字判断）。
    return type(
        original.__name__,
        (original,),
        {"__init__": __init__, "__module__": original.__module__},
    )
