# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: OpenMDW-1.1

"""Optional HCU pure fused operators used by the Wan VAE encoder.

Only the proven, positive-gain Demand 2 (Causal Cache + Pad + Conv3D) is included,
restricted to core channels {160, 320, 640}.
Negative-gain operators (RMSNorm+SiLU and Conv3D+Bias+Add) are intentionally omitted
so Inductor's native cross-node Triton fusion is preserved.
"""

from turbo_physai.optimizations.models.cosmos3.vae.fused_ops._hipdnn import (
    conv_channels_supported,
    fused_dtype_supported,
    fused_ops_enabled,
)
from turbo_physai.optimizations.models.cosmos3.vae.fused_ops.concat_conv_bias import (
    fused_causal_cache_pad_conv3d_ndhwc,
    fused_concat_conv_bias,
)
from turbo_physai.optimizations.models.cosmos3.vae.fused_ops.conv_bias import fused_conv3d_bias_ndhwc

__all__ = [
    "conv_channels_supported",
    "fused_causal_cache_pad_conv3d_ndhwc",
    "fused_concat_conv_bias",
    "fused_conv3d_bias_ndhwc",
    "fused_dtype_supported",
    "fused_ops_enabled",
]
