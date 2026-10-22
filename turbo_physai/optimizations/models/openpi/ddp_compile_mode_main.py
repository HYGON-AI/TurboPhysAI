# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Set torch Dynamo's DDP compile mode before upstream model construction."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
import logging

from .ddp_state import configure_optimize_ddp, register_compile_options


def wrap_init_logging_ddp_compile_mode(original, options: Mapping[str, object]):
    register_compile_options(options)

    @wraps(original)
    def wrapped(*args, **kwargs):
        # Match the reference launcher: this must happen before train_loop
        # creates the model, DDP wrapper, or any compiled checkpoint region.
        configure_optimize_ddp()
        logging.getLogger(__name__).info(
            "Turbo set torch._dynamo.config.optimize_ddp=%s before train logging",
            options.get("optimize_ddp", "no_optimization"),
        )
        return original(*args, **kwargs)

    return wrapped
