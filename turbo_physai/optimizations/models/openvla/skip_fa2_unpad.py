# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""右 padding prefill 时跳过 FA2 的 causal mask 构造，不再走 varlen/unpad。"""

import functools


def _is_right_padded(attention_mask):
    """判断是否右padding或无padding：首列没有 0 """

    return attention_mask.ndim == 2 and bool(attention_mask[:, 0].all().item())


def make_fast_fa2_causal_mask_wrapper(original, options=None):
    """wrap 工厂：FA2 + prefill + 右 padding 时返回 None，跳过 mask 构造。"""

    del options

    @functools.wraps(original)
    def wrapper(self, attention_mask, input_tensor, cache_position, past_seen_tokens):
        if (
            self.config._attn_implementation == "flash_attention_2"
            and attention_mask is not None
            and past_seen_tokens == 0
            and _is_right_padded(attention_mask)
        ):
            return None
        return original(
            self, attention_mask, input_tensor, cache_position, past_seen_tokens
        )

    return wrapper
