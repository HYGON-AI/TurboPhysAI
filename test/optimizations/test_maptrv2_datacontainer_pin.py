# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for maptrv2.datacontainer_pin; simulated pinning is CPU-only."""

import collections
import importlib
import sys
import types

import pytest

from turbo_physai.optimizations.models.maptrv2.datacontainer_pin import (
    datacontainer_pin_wrapper,
)

torch = pytest.importorskip("torch")
PIN_MODULE = "torch.utils.data._utils.pin_memory"
PIN_TARGET = PIN_MODULE + ".pin_memory"


class Container:
    def __init__(self, data, cpu_only=False):
        self._data = data
        self.cpu_only = cpu_only
        self.stack = True
        self.padding_value = 7
        self.pad_dims = 2

    @property
    def data(self):
        return self._data


@pytest.fixture
def pinning(monkeypatch):
    parallel = types.ModuleType("mmcv.parallel")
    parallel.DataContainer = Container
    monkeypatch.setitem(sys.modules, "mmcv.parallel", parallel)
    module = importlib.import_module(PIN_MODULE)
    name = "pin_memory"
    original = getattr(module, name)
    calls = []
    pinned = set()

    def pin(tensor, *args, **kwargs):
        calls.append(tensor)
        pinned.add(id(tensor))
        return tensor

    monkeypatch.setattr(torch.Tensor, "pin_memory", pin)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda t: id(t) in pinned)
    wrapped = datacontainer_pin_wrapper(original, {})
    monkeypatch.setattr(module, name, wrapped)
    return wrapped, original, calls, pinned


def test_nested_containers_and_native_types(pinning):
    pin, original, calls, _ = pinning
    named = collections.namedtuple("Named", "value")
    image, metadata = torch.ones(2), torch.zeros(2)
    dc = Container([named(Container(image)), Container(metadata, cpu_only=True)])
    batch = collections.OrderedDict(img=dc, other="metadata")
    result = pin(batch)
    assert isinstance(result, collections.OrderedDict)
    assert isinstance(result["img"].data[0], named)
    assert result["img"] is dc
    assert dc.stack and dc.padding_value == 7 and dc.pad_dims == 2
    assert len(calls) == 1 and calls[0] is image
    assert pin.__wrapped__ is original


def test_already_pinned_and_device_tensors_are_untouched(pinning):
    pin, _, calls, pinned = pinning
    cpu = torch.ones(1)
    pinned.add(id(cpu))
    device = torch.empty(1, device="meta")
    assert pin(Container(cpu)).data is cpu
    assert pin(Container(device)).data is device
    assert not calls


def test_duplicate_install_is_idempotent(pinning):
    pin, _, _, _ = pinning
    assert datacontainer_pin_wrapper(pin, {}) is pin


def test_reference_patch_is_not_wrapped_twice():
    namespace = {"_pin_recursive": lambda x: x}
    exec("def _pin_memory_with_data_container(data): return data", namespace)
    reference = namespace["_pin_memory_with_data_container"]
    assert datacontainer_pin_wrapper(reference, {}) is reference


def test_engine_restores_native_function(pinning):
    from turbo_physai.engine.contracts import Mechanism, ReplacementSpec
    from turbo_physai.engine.execution.replacements.wrapper import WrapperHandler

    wrapped, original, _, _ = pinning
    module = importlib.import_module(PIN_MODULE)
    name = "pin_memory"
    setattr(module, name, original)
    handler = WrapperHandler()
    spec = ReplacementSpec(
        replacement_id="test.pin", mechanism=Mechanism.WRAPPER,
        target=PIN_TARGET,
        replacement=datacontainer_pin_wrapper.__module__ + ".datacontainer_pin_wrapper",
    )
    prepared = handler.prepare(spec, {})
    snapshot = handler.snapshot(prepared)
    try:
        handler.apply(prepared)
        assert getattr(module, name).__turbo_physai_datacontainer_pin__
        dc = Container(torch.ones(1))
        getattr(module, name)({"img": dc})
    finally:
        handler.restore(snapshot)
    assert getattr(module, name) is original


def _real_collate(batch):
    from mmcv.parallel import DataContainer
    return {"img": DataContainer(torch.stack(batch)),
            "meta": DataContainer(torch.ones(1), cpu_only=True)}


@pytest.mark.model_deps
def test_real_mmcv_container_properties_with_simulated_allocator(monkeypatch):
    parallel = pytest.importorskip("mmcv.parallel")
    module = importlib.import_module(PIN_MODULE)
    name = "pin_memory"
    calls = []
    monkeypatch.setattr(torch.Tensor, "pin_memory",
                        lambda tensor, *args, **kwargs: calls.append(tensor) or tensor)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda tensor: False)
    monkeypatch.setattr(module, name, datacontainer_pin_wrapper(getattr(module, name), {}))
    data = torch.ones(2)
    container = parallel.DataContainer([data], stack=True, padding_value=9, pad_dims=1)
    metadata = parallel.DataContainer(torch.zeros(2), cpu_only=True)
    result = getattr(module, name)({"img": container, "meta": metadata})
    assert result["img"] is container and result["meta"] is metadata
    assert container.stack and container.padding_value == 9 and container.pad_dims == 1
    assert container.data[0] is data
    assert len(calls) == 1 and calls[0] is data


@pytest.mark.hcu
@pytest.mark.model_deps
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_loader_epochs(monkeypatch, workers, enabled):
    pytest.importorskip("mmcv.parallel")
    if not torch.cuda.is_available():
        pytest.skip("device-backed pinned allocator required")
    module = importlib.import_module("torch.utils.data._utils.pin_memory")
    name = "pin_memory"
    monkeypatch.setattr(module, name, datacontainer_pin_wrapper(getattr(module, name), {}))
    loader = torch.utils.data.DataLoader(
        torch.arange(16).reshape(8, 2), batch_size=2, num_workers=workers,
        pin_memory=enabled, collate_fn=_real_collate,
        persistent_workers=workers > 0,
    )
    for _ in range(2):
        for batch in loader:
            assert batch["img"].data.is_pinned() == enabled
            assert not batch["meta"].data.is_pinned()
    del loader
