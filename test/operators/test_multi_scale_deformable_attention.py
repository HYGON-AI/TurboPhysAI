# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import types

import pytest


from turbo_physai.operators import multi_scale_deformable_attention as operator


def test_lightop_available_requires_both_functions(monkeypatch):
    monkeypatch.setattr(operator, "_lightop", None)
    assert operator.lightop_available() is False

    monkeypatch.setattr(
        operator,
        "_lightop",
        types.SimpleNamespace(ms_deform_attn_forward=lambda *args: None),
    )
    assert operator.lightop_available() is False

    monkeypatch.setattr(
        operator,
        "_lightop",
        types.SimpleNamespace(
            ms_deform_attn_forward=lambda *args: None,
            ms_deform_attn_backward=lambda *args: None,
        ),
    )
    assert operator.lightop_available() is True


def test_forward_delegates_to_lightop(monkeypatch):
    calls = []
    marker = object()

    class LightOp:
        @staticmethod
        def ms_deform_attn_forward(*args):
            calls.append(args)
            return marker

        @staticmethod
        def ms_deform_attn_backward(*args):
            return None

    monkeypatch.setattr(operator, "_lightop", LightOp())
    args = ("value", "shapes", "starts", "locations", "weights", 64)

    assert operator.ms_deform_attn_forward(*args) is marker
    assert calls == [args]


def test_backward_makes_grad_output_contiguous_and_delegates(monkeypatch):
    calls = []

    class NonContiguousGradient:
        def is_contiguous(self):
            return False

        def contiguous(self):
            return "contiguous-gradient"

    class LightOp:
        @staticmethod
        def ms_deform_attn_forward(*args):
            return None

        @staticmethod
        def ms_deform_attn_backward(*args):
            calls.append(args)

    monkeypatch.setattr(operator, "_lightop", LightOp())
    args = (
        "value",
        "shapes",
        "starts",
        "locations",
        "weights",
        NonContiguousGradient(),
        "grad-value",
        "grad-locations",
        "grad-weights",
        64,
    )

    assert operator.ms_deform_attn_backward(*args) is None
    assert calls[0][5] == "contiguous-gradient"
    assert calls[0][1:5] == args[1:5]
    assert calls[0][6:] == args[6:]


def test_missing_lightop_raises_runtime_error(monkeypatch):
    monkeypatch.setattr(operator, "_lightop", None)

    with pytest.raises(RuntimeError, match="lightop is not installed"):
        operator.ms_deform_attn_forward(None, None, None, None, None, 64)

