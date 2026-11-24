# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Avoid averaging attention weights over heads."""

import functools


def attention_no_average_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, query, key, value, *args, **kwargs):
        if len(args) < 4 and 'average_attn_weights' not in kwargs:
            kwargs['average_attn_weights'] = False
        return original(self, query, key, value, *args, **kwargs)

    return wrapped
