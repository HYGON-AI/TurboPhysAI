# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""HIP hardware-queue sizing so side-stream collectives can overlap compute.

DTK HIP maps streams round-robin onto ``GPU_MAX_HW_QUEUES`` hardware queues
(default 4) at first use. Two streams sharing a queue execute serially, so a
ZeRO reduction or RCCL stream landing on the compute stream's queue removes
all communication/compute overlap even though the streams are distinct.
The queue count does not change stream priorities.
"""

import logging
import os

from turbo_physai.optimizations.models.openwam.runtime import integer
from turbo_physai.optimizations.models.openwam.settings import value

logger = logging.getLogger(__name__)


def requested():
    """Queue count from the training Group options; zero keeps the default."""
    return integer("hw_queues", 0, minimum=2) if value("hw_queues", 0) else 0


def configure():
    """Must run before the HIP runtime initializes; an explicit GPU_MAX_HW_QUEUES wins."""
    count = requested()
    if not count:
        return
    import torch

    if not getattr(torch.version, "hip", None):
        raise RuntimeError("openwam.training.hw_queues requires a HIP PyTorch build")
    current = os.environ.get("GPU_MAX_HW_QUEUES")
    if current is None:
        if torch.cuda.is_initialized():
            raise RuntimeError(
                "Configure openwam.training.hw_queues before the HIP runtime initializes"
            )
        current = os.environ.setdefault("GPU_MAX_HW_QUEUES", str(count))
    if current != str(count):
        logger.warning(
            "[optimizations] %s=%d ignored: GPU_MAX_HW_QUEUES=%s already set",
            "hw_queues",
            count,
            current,
        )
