# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import subprocess
import sys
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")

from turbo_physai.optimizations.models.dreamzero import (
    CausalWanAttentionBlock,
    CausalWanSelfAttention,
    text_encoder,
    wan_video_image_encoder,
    wan_video_vae,
)


def test_loading_state_block_fold_does_not_construct_compiled_functions():
    subprocess.run(
        [sys.executable, "-c", """
import os
from unittest.mock import patch

import torch

os.environ.pop("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", None)
with patch.object(torch, "compile") as compile:
    from turbo_physai.optimizations.models.dreamzero.CausalWanSelfAttention import process_state_blocks
    from turbo_physai.optimizations.models.dreamzero import CausalWanAttentionBlock
    compile.assert_not_called()
"""],
        check=True,
    )


@pytest.mark.parametrize(
    "factory, eager, compile_count",
    [
        (CausalWanSelfAttention.compile_forward, CausalWanSelfAttention.forward, 2),
        (CausalWanAttentionBlock.compile_forward, CausalWanAttentionBlock.forward, 2),
        (wan_video_vae.compile_encoder3d_forward, None, 1),
        (wan_video_vae.compile_encoder3d_38_forward, None, 1),
        (text_encoder.compile_forward, None, 1),
        (wan_video_image_encoder.compile_forward, None, 1),
    ],
)
def test_compile_group_factories_use_the_same_switch(monkeypatch, factory, eager, compile_count):
    original = Mock()
    compile = Mock(side_effect=lambda function, **options: function)
    monkeypatch.setattr(torch, "compile", compile)
    options = {"backend": "eager"}

    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    assert factory(original, options) is (original if eager is None else eager)
    compile.assert_not_called()

    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0")
    assert factory(original, options).__wrapped__ is original
    assert compile.call_count == compile_count
    assert all(call.kwargs["backend"] == "eager" for call in compile.call_args_list)


def test_compile_switch_skips_cross_attention_compilation(monkeypatch):
    monkeypatch.setenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "1")
    original = Mock()
    assert CausalWanAttentionBlock.compile_cross_attn(original, {}) is original
