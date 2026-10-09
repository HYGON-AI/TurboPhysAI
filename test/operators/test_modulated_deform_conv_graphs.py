# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import types

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("hipdnn")

from turbo_physai.operators import modulated_deform_conv as operator


class _Node:
    def __init__(self, name):
        self.name = name
        self.output = False
        self.data_type = None
        self.dim = None

    def set_output(self, value):
        self.output = value
        return self

    def set_data_type(self, value):
        self.data_type = value
        return self

    def set_dim(self, value):
        self.dim = value
        return self


class _Graph:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.calls = []
        self.outputs = {}

    def tensor_like(self, tensor):
        return _Node(f"tensor-{len(self.calls)}")

    def deform_conv_fprop(self, **kwargs):
        self.calls.append(("fprop", kwargs))
        out = _Node("out")
        self.outputs["out"] = out
        return out

    def deform_conv_wgrad(self, **kwargs):
        self.calls.append(("wgrad", kwargs))
        dw = _Node("dw")
        self.outputs["dw"] = dw
        return dw

    def deform_conv_dgrad(self, **kwargs):
        self.calls.append(("dgrad", kwargs))
        dx, doffset, dmask = _Node("dx"), _Node("doffset"), _Node("dmask")
        self.outputs.update(dx=dx, doffset=doffset, dmask=dmask)
        return dx, doffset, dmask

    def validate(self):
        self.calls.append(("validate", {}))

    def build_operation_graph(self):
        self.calls.append(("build_operation_graph", {}))

    def create_execution_plans(self):
        self.calls.append(("create_execution_plans", {}))

    def check_support(self):
        self.calls.append(("check_support", {}))

    def build_plans(self):
        self.calls.append(("build_plans", {}))


def _install_fake_hipdnn(monkeypatch):
    graphs = []

    fake = types.SimpleNamespace(
        data_type=types.SimpleNamespace(FLOAT="FLOAT", HALF="HALF"),
        pygraph=lambda name, **kwargs: graphs.append(_Graph(name, **kwargs))
        or graphs[-1],
    )
    monkeypatch.setattr(operator, "hipdnn", fake)
    return graphs


def test_build_fprop_graph_sets_output(monkeypatch):
    graphs = _install_fake_hipdnn(monkeypatch)
    tensors = [torch.zeros(1, dtype=torch.float32) for _ in range(4)]

    graph, *_rest, out = operator.build_fprop_graph(
        *tensors, (1, 1), (1, 1), (1, 1)
    )

    assert graphs == [graph]
    assert graph.name == "deform_convolution"
    assert out.output is True
    assert out.data_type == "FLOAT"
    assert graph.calls[0][0] == "fprop"


def test_autograd_function_is_a_torch_function():
    assert issubclass(
        operator.ModulatedDeformConv2dFunction,
        torch.autograd.Function,
    )


def test_build_wrw_graph_sets_weight_dimension(monkeypatch):
    graphs = _install_fake_hipdnn(monkeypatch)
    input, offset, weight, mask, grad_output = (
        torch.zeros(1, dtype=torch.float32) for _ in range(5)
    )
    weight = torch.zeros(4, 8, 3, 3, dtype=torch.float32)

    graph, *_rest, dw = operator.build_wrw_graph(
        input, offset, weight, mask, grad_output, (1, 1), (1, 1), (1, 1)
    )

    assert graphs == [graph]
    assert graph.name == "deform_convolution_wrw"
    assert dw.output is True
    assert dw.dim == weight.shape
    assert dw.data_type == "FLOAT"
    assert graph.calls[0][0] == "wgrad"


def test_build_dx_graph_outputs_only_required_gradients(monkeypatch):
    graphs = _install_fake_hipdnn(monkeypatch)
    input, offset, weight, mask, grad_output = (
        torch.zeros(1, dtype=torch.float32) for _ in range(5)
    )

    graph, *nodes = operator.build_dx_graph(
        input,
        offset,
        weight,
        mask,
        grad_output,
        (1, 1),
        (1, 1),
        (1, 1),
        (True, False, True),
    )

    dx, doffset, dmask = nodes[-3:]
    assert graphs == [graph]
    assert graph.name == "deform_convolution_bwd"
    assert dx.output is True
    assert doffset.output is False
    assert dmask.output is True
    assert graph.calls[0][0] == "dgrad"
