# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
apply_rope 的 sin/cos 表缓存。
"""

from __future__ import annotations

import torch

_state = {"pos": None, "version": -1, "d": 0, "wl": 0.0, "dtype": None, "cos": None, "sin": None}


@torch._dynamo.disable
def _rope_tables(positions, d, max_wavelength, dtype):
    state = _state
    if (
        state["pos"] is positions
        and state["version"] == positions._version
        and state["d"] == d
        and state["wl"] == max_wavelength
        and state["dtype"] == dtype
        and state["cos"] is not None
        and state["cos"].device == positions.device
    ):
        return state["cos"], state["sin"]

    d_half = d // 2
    freq_exponents = (2.0 / d) * torch.arange(d_half, dtype=dtype, device=positions.device)
    timescale = max_wavelength**freq_exponents
    radians = torch.einsum("bl,h->blh", positions.to(dtype), 1.0 / timescale)
    radians = radians[..., None, :]  # [B, L, 1, D/2]

    state.update(pos=positions, version=positions._version, d=d, wl=max_wavelength,
                 dtype=dtype, cos=torch.cos(radians), sin=torch.sin(radians))
    return state["cos"], state["sin"]


def apply_rope(x, positions, max_wavelength: float = 10_000.0, dtype: torch.dtype = torch.float32):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    original_dtype = x.dtype
    d_half = x.shape[-1] // 2

    cos, sin = _rope_tables(positions, x.shape[-1], max_wavelength, dtype)

    x_casted = x.to(dtype)
    x1, x2 = x_casted.split(d_half, dim=-1)
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    return res.to(original_dtype)
