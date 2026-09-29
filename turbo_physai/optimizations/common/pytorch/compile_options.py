# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch


def configure_compile_cache(
    *,
    recompile_limit: int,
    cache_size_limit: int,
    accumulated_cache_size_limit: int,
    use_duck_shape: bool,
) -> None:
    """Set caller-selected Dynamo/FX options when the installed version exposes them.

    Model-specific thresholds and defaults belong to the model adapter.
    This function is inert until explicitly called.
    """
    try:
        torch._dynamo.config.recompile_limit = recompile_limit
    except AttributeError:
        pass
    try:
        torch._dynamo.config.cache_size_limit = cache_size_limit
    except AttributeError:
        pass
    try:
        torch._dynamo.config.accumulated_cache_size_limit = accumulated_cache_size_limit
    except AttributeError:
        pass
    try:
        torch.fx.experimental._config.use_duck_shape = use_duck_shape
    except (AttributeError, ModuleNotFoundError):
        pass
