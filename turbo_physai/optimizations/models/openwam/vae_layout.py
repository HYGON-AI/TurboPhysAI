# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Keep native VAE pointwise producers in the convolution's physical layout.

``contiguous`` alone can disappear inside a fused Inductor graph. The stride
constraint fixes the producer's layout before cache extraction / causal padding;
it does not change the causal padding or materialize an extra eager clone when
the tensor already has the required layout.
"""

import torch


def validate():
    try:
        from torch._inductor.inductor_prims import force_stride_order
    except ImportError as exc:
        raise RuntimeError(
            "VAE layout repair requires Inductor force_stride_order"
        ) from exc
    if not callable(force_stride_order):
        raise RuntimeError("Inductor force_stride_order is unavailable")


def channels_last(x):
    if x.ndim != 5:
        return x
    if not torch.compiler.is_compiling():
        return x.contiguous(memory_format=torch.channels_last_3d)
    from torch._inductor.inductor_prims import force_stride_order

    _, c, t, h, w = x.shape
    return force_stride_order(x, [c * t * h * w, 1, h * w * c, w * c, c])
