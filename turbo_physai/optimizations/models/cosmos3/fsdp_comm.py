# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Model-side access to the ordered FSDP collectives.

When the adapted Cosmos tree provides ``cosmos_framework.utils.fsdp_ordered_comm``
its stream table is reused so every collective (FSDP all-gather/reduce-scatter,
DTensor norm reduction, loss all-reduce) shares one canonical stream per process
group.  On a clean upstream tree the TurboPhysAI common implementation is used.
"""

try:
    from cosmos_framework.utils.fsdp_ordered_comm import (
        OrderedAllGather,
        OrderedReduceScatter,
        materialize_dtensor_norm,
        ordered_all_reduce,
    )
except ModuleNotFoundError as exc:
    if exc.name != "cosmos_framework.utils.fsdp_ordered_comm":
        raise
    from ...common.pytorch.fsdp_ordered_comm import (
        OrderedAllGather,
        OrderedReduceScatter,
        materialize_dtensor_norm,
        ordered_all_reduce,
    )

__all__ = [
    "OrderedAllGather",
    "OrderedReduceScatter",
    "materialize_dtensor_norm",
    "ordered_all_reduce",
]
