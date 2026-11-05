# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA spawn-worker DataLoader。

把 RLDS（TFDS 支撑的）训练数据管线挪进 spawn 出来的 DataLoader worker：
worker必须用 spawn（fork 一个 TF 线程池活跃的进程会死锁），而 spawn 要求
dataset 可 pickle，所以数据集类只序列化构造参数、在 worker 里重建 TF graph。

只处理 RLDS 数据集构造的 DataLoader：注入 ``NUM_WORKERS``、spawn context 和
``pin_memory``。其他 DataLoader（eval、非 RLDS、HF 内部）保持上游行为。
"""

from __future__ import annotations

import importlib

NUM_WORKERS = 2
PIN_MEMORY = True


def _is_rlds(dataset) -> bool:
    cls = type(dataset)
    if getattr(cls, "__turbo_physai_rlds_dataset__", False):
        return True
    return cls.__module__ == "prismatic.vla.datasets.datasets" and cls.__name__ in (
        "RLDSDataset",
        "EpisodicRLDSDataset",
    )


def _identity(subclass: type, original: type) -> type:
    subclass.__name__ = original.__name__
    subclass.__qualname__ = original.__qualname__
    return subclass


def _picklable_dataset(original: type) -> type:
    """只序列化构造参数、在 worker 里重建 TF graph 的数据集子类。"""

    module, qualname = original.__module__, original.__qualname__

    class Picklable(original):
        def __init__(self, *args, **kwargs):
            self._ctor_args, self._ctor_kwargs = args, kwargs
            super().__init__(*args, **kwargs)

        def __reduce__(self):
            return _reconstruct, (module, qualname, self._ctor_args, self._ctor_kwargs)

    Picklable.__turbo_physai_rlds_dataset__ = True
    return _identity(Picklable, original)


def _reconstruct(module, qualname, args, kwargs):
    """在 worker 内按类路径重建数据集（TF graph 在构造函数里重建）。"""

    target = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)

    from .reproducibility import worker_seeded_dataset_class

    return worker_seeded_dataset_class(target)(*args, **kwargs)


def rlds_dataset_spawn_wrapper(original, options=None):
    del options
    return _picklable_dataset(original)


def episodic_rlds_dataset_spawn_wrapper(original, options=None):
    del options
    return _picklable_dataset(original)


def dataloader_spawn_wrapper(original, options=None):
    """只对 RLDS 数据集注入 num_workers / spawn context / pin_memory。"""

    del options

    class SpawnDataLoader(original):
        def __init__(self, dataset, *args, **kwargs):
            if _is_rlds(dataset):
                kwargs["num_workers"] = NUM_WORKERS
                kwargs["multiprocessing_context"] = _spawn_context()
                if PIN_MEMORY:
                    kwargs["pin_memory"] = True
            super().__init__(dataset, *args, **kwargs)

    return _identity(SpawnDataLoader, original)


_CONTEXT = None


def _spawn_context():
    global _CONTEXT
    if _CONTEXT is None:
        import torch.multiprocessing as mp

        _CONTEXT = mp.get_context("spawn")
    return _CONTEXT
