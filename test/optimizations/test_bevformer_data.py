# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import random
import sys
import types

import numpy as np
import pytest


torch = pytest.importorskip("torch")


class _DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False,
                 pad_dims=None):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims


def _install_framework_stubs():
    """Expose only the public interfaces imported by the replacement module."""

    mmcv = types.ModuleType("mmcv")
    parallel = types.ModuleType("mmcv.parallel")
    data_container = types.ModuleType("mmcv.parallel.data_container")
    runner = types.ModuleType("mmcv.runner")
    parallel.collate = lambda batch, **kwargs: batch
    parallel.DataContainer = _DataContainer
    data_container.DataContainer = _DataContainer
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

    modules = {
        "mmcv": mmcv,
        "mmcv.parallel": parallel,
        "mmcv.parallel.data_container": data_container,
        "mmcv.runner": runner,
        "mmdet": mmdet,
        "mmdet.datasets": datasets,
        "mmdet.datasets.samplers": samplers,
    }
    for name, module in modules.items():
        sys.modules.setdefault(name, module)


_install_framework_stubs()

from turbo_physai.optimizations.models.bevformer import data


def test_worker_init_fn_sets_reproducible_numpy_and_python_seeds():
    data.worker_init_fn(worker_id=2, num_workers=4, rank=3, seed=7)
    actual = (np.random.rand(), random.random())
    np.random.seed(21)
    random.seed(21)
    assert actual == pytest.approx((np.random.rand(), random.random()))


def test_cuda_prefetch_metadata_group_discovery():
    loader = object.__new__(data.CudaPrefetchLoader)
    first = {"lidar2img": [np.eye(4)], "img_shape": [(10, 10, 3)]}
    second = {"lidar2img": [np.eye(4)], "img_shape": [(10, 10, 3)]}
    assert loader._collect_meta_groups([[first], [second]]) == [[first, second]]
    assert loader._collect_meta_groups({"unrelated": [1, 2]}) == []


def test_build_dataloader_non_distributed_contract(monkeypatch):
    captured = {}

    class Loader:
        def __init__(self, dataset, **kwargs):
            captured["dataset"] = dataset
            captured.update(kwargs)

    dataset = object()
    projects = types.ModuleType("projects")
    plugin = types.ModuleType("projects.mmdet3d_plugin")
    datasets = types.ModuleType("projects.mmdet3d_plugin.datasets")
    samplers = types.ModuleType("projects.mmdet3d_plugin.datasets.samplers")
    sampler = types.ModuleType(
        "projects.mmdet3d_plugin.datasets.samplers.sampler"
    )
    sampler.build_sampler = lambda *args, **kwargs: None
    for name, module in {
        "projects": projects,
        "projects.mmdet3d_plugin": plugin,
        "projects.mmdet3d_plugin.datasets": datasets,
        "projects.mmdet3d_plugin.datasets.samplers": samplers,
        "projects.mmdet3d_plugin.datasets.samplers.sampler": sampler,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(data, "DataLoader", Loader)
    monkeypatch.setattr(data, "get_dist_info", lambda: (0, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = data.build_dataloader(
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
    assert captured["batch_size"] == 4
    assert captured["num_workers"] == 6
    assert captured["sampler"] is None
    assert captured["pin_memory"] is True
    assert captured["prefetch_factor"] == 16
    assert captured["persistent_workers"] is True
