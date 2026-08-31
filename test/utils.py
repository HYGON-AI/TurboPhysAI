# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
import os
import pytest
import torch
import numpy as np
import time
from functools import partial

def run_benchmark(func, repeat_num, backward=False, grad=None, warmup=10):
    assert not backward or grad is not None, "backward=True 时必须传入 grad"
    assert isinstance(func, partial), "func 必须是 functools.partial 对象，以便提取其中的 tensor 参数进行 grad 清零"

    partial_args = func.args
    partial_kwargs = func.keywords or {}

    all_tensors = []
    for arg in partial_args:
        if isinstance(arg, torch.Tensor):
            all_tensors.append(arg)
    for kwarg in partial_kwargs.values():
        if isinstance(kwarg, torch.Tensor):
            all_tensors.append(kwarg)

    for i in range(warmup):
        output = func()
        if backward:
            output.backward(grad)

    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(repeat_num):
        if backward:
            for tensor in all_tensors:
                tensor.grad = None

        output = func()
        if backward:
            output.backward(grad)

    torch.cuda.synchronize()
    avg_ms = (time.time() - start_time) * 1000 / repeat_num
    return avg_ms, output


def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    assert torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan), \
        f"data not close, atol: {(a - b).abs().max()}, rtol: {(a - b).abs().max() / (b.abs().max() + 1e-8)}"
