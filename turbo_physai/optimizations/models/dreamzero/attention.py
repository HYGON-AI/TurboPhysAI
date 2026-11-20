"""Use FA2's dense entry point for uniform DiT cross-attention batches."""

import functools
import os
import warnings


def dense_flash_attention(original, options):
    from groot.vla.model.dreamzero.modules import attention

    del options

    @functools.wraps(original)
    def flash_attention(
        q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
        q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
        dtype=attention.torch.bfloat16, version=None,
    ):
        use_fa3 = (version is None or version == 3) and attention.FLASH_ATTN_3_AVAILABLE
        if (
            q_lens is None and k_lens is None
            and attention.FLASH_ATTN_2_AVAILABLE and not use_fa3
            and os.getenv("ENABLE_TENSORRT", "False").lower() != "true"
            and attention._gpu_supports_flash_attention()
        ):
            half_dtypes = (attention.torch.float16, attention.torch.bfloat16)
            assert dtype in half_dtypes
            assert q.device.type == "cuda" and q.size(-1) <= 256
            if version == 3:
                warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.")
            out_dtype = q.dtype
            q, k, v = (x if x.dtype in half_dtypes else x.to(dtype) for x in (q, k, v))
            q, k = q.to(v.dtype), k.to(v.dtype)
            if q_scale is not None:
                q = q * q_scale
            return attention.flash_attn.flash_attn_func(
                q, k, v, dropout_p=dropout_p, softmax_scale=softmax_scale,
                causal=causal, window_size=window_size, deterministic=deterministic,
            ).to(out_dtype)
        return original(
            q, k, v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
            softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
            window_size=window_size, deterministic=deterministic, dtype=dtype,
            version=version,
        )

    return flash_attention
