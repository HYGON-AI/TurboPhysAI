# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the MapTRv2 DataLoader worker context."""

from __future__ import annotations

import importlib
import sys
import types

import pytest


torch = pytest.importorskip("torch")


def _install_framework_stubs(monkeypatch):
    mmcv = types.ModuleType("mmcv")
    parallel = types.ModuleType("mmcv.parallel")
    runner = types.ModuleType("mmcv.runner")
    parallel.collate = lambda batch, **kwargs: batch
    runner.get_dist_info = lambda: (0, 1)
    mmcv.parallel = parallel
    mmcv.runner = runner

    mmdet = types.ModuleType("mmdet")
    datasets = types.ModuleType("mmdet.datasets")
    samplers = types.ModuleType("mmdet.datasets.samplers")

    class GroupSampler:
        def __init__(self, dataset, samples_per_gpu):
            self.dataset = dataset
            self.samples_per_gpu = samples_per_gpu

    samplers.GroupSampler = GroupSampler
    datasets.samplers = samplers
    mmdet.datasets = datasets

    projects = types.ModuleType("projects")
    plugin = types.ModuleType("projects.mmdet3d_plugin")
    project_datasets = types.ModuleType("projects.mmdet3d_plugin.datasets")
    project_samplers = types.ModuleType(
        "projects.mmdet3d_plugin.datasets.samplers"
    )
    distributed_sampler = types.ModuleType(
        "projects.mmdet3d_plugin.datasets.samplers.distributed_sampler"
    )
    group_sampler = types.ModuleType(
        "projects.mmdet3d_plugin.datasets.samplers.group_sampler"
    )
    sampler = types.ModuleType(
        "projects.mmdet3d_plugin.datasets.samplers.sampler"
    )
    distributed_sampler.DistributedSampler = type("DistributedSampler", (), {})
    group_sampler.DistributedGroupSampler = type(
        "DistributedGroupSampler", (), {}
    )
    sampler.build_sampler = lambda *args, **kwargs: None
    project_datasets.samplers = project_samplers
    project_samplers.distributed_sampler = distributed_sampler
    project_samplers.group_sampler = group_sampler
    project_samplers.sampler = sampler
    plugin.datasets = project_datasets
    projects.mmdet3d_plugin = plugin

    modules = {
        "mmcv": mmcv,
        "mmcv.parallel": parallel,
        "mmcv.runner": runner,
        "mmdet": mmdet,
        "mmdet.datasets": datasets,
        "mmdet.datasets.samplers": samplers,
        "projects": projects,
        "projects.mmdet3d_plugin": plugin,
        "projects.mmdet3d_plugin.datasets": project_datasets,
        "projects.mmdet3d_plugin.datasets.samplers": project_samplers,
    }
    modules[
        "projects.mmdet3d_plugin.datasets.samplers.distributed_sampler"
    ] = distributed_sampler
    modules[
        "projects.mmdet3d_plugin.datasets.samplers.group_sampler"
    ] = group_sampler
    modules["projects.mmdet3d_plugin.datasets.samplers.sampler"] = sampler
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def maptrv2_data(monkeypatch):
    module_name = (
        "turbo_physai.optimizations.models.maptrv2.data"
    )
    _install_framework_stubs(monkeypatch)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    monkeypatch.delitem(sys.modules, module_name, raising=False)


def _build_loader_capture(maptrv2_data, monkeypatch):
    captured = {}

    class Loader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured.update(kwargs)

    monkeypatch.setattr(maptrv2_data, "DataLoader", Loader)
    monkeypatch.setattr(maptrv2_data, "get_dist_info", lambda: (0, 1))
    dataset = object()
    loader = maptrv2_data.build_dataloader(
        dataset,
        samples_per_gpu=2,
        workers_per_gpu=3,
        num_gpus=2,
        dist=False,
        shuffle=False,
        seed=11,
    )
    assert isinstance(loader, Loader)
    assert captured["dataset"] is dataset
    return captured


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork is unavailable on Windows"
)
def test_dataloader_defaults_to_fork_context(maptrv2_data, monkeypatch):
    monkeypatch.delenv("TURBO_PHYSAI_FORK_START_METHOD", raising=False)
    monkeypatch.delenv("TURBO_PHYSAI_DATALOADER_START_METHOD", raising=False)

    captured = _build_loader_capture(maptrv2_data, monkeypatch)

    assert captured["num_workers"] == 6
    assert captured["pin_memory"] is True
    assert captured["multiprocessing_context"].get_start_method() == "fork"


def test_dataloader_context_can_be_overridden(maptrv2_data, monkeypatch):
    monkeypatch.setenv("TURBO_PHYSAI_FORK_START_METHOD", "1")
    monkeypatch.setenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "spawn")

    captured = _build_loader_capture(maptrv2_data, monkeypatch)

    assert captured["multiprocessing_context"].get_start_method() == "spawn"


def test_dataloader_context_can_be_disabled(maptrv2_data, monkeypatch):
    monkeypatch.setenv("TURBO_PHYSAI_FORK_START_METHOD", "0")

    captured = _build_loader_capture(maptrv2_data, monkeypatch)

    assert "multiprocessing_context" not in captured
