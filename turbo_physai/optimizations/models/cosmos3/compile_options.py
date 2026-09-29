# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from ...common.pytorch.compile_options import configure_compile_cache


def set_torch_compile_options(recompile_limit: int = 64, use_duck_shape: bool = False):
    """Apply the existing Cosmos3 compile-cache policy."""
    effective_limit = max(recompile_limit, 64) if recompile_limit else 64
    cache_limit = max(effective_limit, 512)
    configure_compile_cache(
        recompile_limit=effective_limit,
        cache_size_limit=cache_limit,
        accumulated_cache_size_limit=max(cache_limit * 8, 4096),
        use_duck_shape=use_duck_shape,
    )
