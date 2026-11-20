"""Real-valued (cos/sin) RoPE math, standing in for the complex/polar RoPE used
by the target model's rope_params/rope_action_apply/causal_rope_action_apply.

Real-valued Q/K rotation avoids Inductor's complex-operator fallback and lets
CausalWanSelfAttention compile its RoPE tensor region efficiently.
"""

from __future__ import annotations

import torch


def rope_params_real(max_seq_len: int, dim: int, theta: float = 10000) -> torch.Tensor:
    """Build the RoPE multiplier table as interleaved cosine/sine pairs."""
    assert dim % 2 == 0
    angles = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)


def rope_apply_real(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent channels without introducing complex tensors into compile graphs."""
    pairs = x.to(freqs.dtype).unflatten(-1, (-1, 2))
    multipliers = freqs.reshape(1, freqs.shape[0], 1, -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    cos, sin = multipliers[..., 0], multipliers[..., 1]
    return torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1).flatten(-2)


def rope_qk_apply_real(q, k, freqs, output_dtype):
    """Fuse Q/K rotations and their output casts in one tensor-only compile region."""
    return (
        rope_apply_real(q, freqs).to(output_dtype),
        rope_apply_real(k, freqs).to(output_dtype),
    )
