# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA spawn-worker DataLoader replacement for TurboPhysAI.

Why this exists
---------------
OpenVLA's VLA training loop (``TrainingStrategy.run_vla_training``) feeds an
RLDS (TFDS-backed) dataset through a ``DataLoader``.  The baseline creates that
loader with ``num_workers=0``: the (CPU-heavy, Python) batch transform
``RLDSBatchTransform`` — image decode/resize + tokenization + label
construction — runs inline in the training process.

The validated HCU optimization moves the RLDS data pipeline into **spawned**
DataLoader worker processes instead:

* ``num_workers=N`` workers run the batch transform off the main process;
* workers must use the ``spawn`` multiprocessing context: RLDS keeps a
  TensorFlow graph/threadpool alive in the parent, and ``fork``-ing a process
  with an active TF threadpool deadlocks.  ``spawn`` starts a clean process in
  which the TF graph is rebuilt from scratch;
* that requires the dataset instance to be picklable, so the wrapped dataset
  classes serialize their *constructor arguments* instead of the TF graph and
  rebuild the graph inside the worker.

Scope of this Group
-------------------
Only ``DataLoader`` constructions whose dataset is an RLDS dataset are touched:
they get ``num_workers=N`` (Group ``options`` key ``num_workers``, default ``1``),
the ``spawn`` context, and — when ``N > 0`` — ``pin_memory=True`` (Group
``options`` key ``pin_memory``, default ``True``).  Every other DataLoader in the
process (eval loaders, non-RLDS training, HF internals, ...) keeps the exact
upstream behaviour, and so does any RLDS loader when ``num_workers`` is ``0``.

``pin_memory=True`` is the upstream-HCU item carried by this Group: the pinned
staging buffer is filled by the worker while the main process computes, so the
main process' H2D copy reads from page-locked host memory instead of going
through a pageable copy.  With ``num_workers=0`` there is no worker to overlap
with, so the option is left untouched there (baseline default ``False``).

Three ``wrap`` members compose the Group:

1. ``RLDSDataset``      -> spawn-pickling subclass (constructor-arg serialisation)
2. ``EpisodicRLDSDataset`` -> ditto
3. ``torch.utils.data.DataLoader`` -> subclass that forces ``num_workers=N`` +
   ``spawn`` + ``pin_memory=True`` only when the dataset is an RLDS dataset (also
   patches the early ``from torch.utils.data import DataLoader`` binding in
   ``prismatic.training.strategies.base_strategy`` via ``aliases``).

Module import is side-effect free and does not import torch/transformers/prismatic.

Interaction with ``openvla.reproducibility.*``
---------------------------------------------
Workers rebuild the dataset by re-running its constructor, which happens *before*
``worker_init_fn`` -- too early for anything worker-id dependent.  So the
per-worker data stream seed is installed from the rebuild path itself: the
reconstructor asks ``reproducibility.worker_seeded_dataset_class`` for the class
to instantiate, and that helper returns the original class unchanged unless one
of the reproducibility Groups published a run seed in the parent process (the
marker travels to the worker through the environment).  This Group therefore
behaves exactly as before when those Groups are disabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Optional


# --- helpers -----------------------------------------------------------------

def _rlds_num_workers(options: Optional[Mapping[str, Any]]) -> int:
    """Read the ``num_workers`` Group option (default 1, >= 0)."""
    options = dict(options or {})
    try:
        workers = int(options.get("num_workers", 1))
    except (TypeError, ValueError):
        workers = 1
    return max(0, workers)


def _option_enabled(value: Any, default: bool) -> bool:
    """Parse a boolean Group option (YAML bool, or 1/0, true/false, yes/no, on/off)."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _rlds_pin_memory(options: Optional[Mapping[str, Any]]) -> bool:
    """Read the ``pin_memory`` Group option (default ``True``)."""
    options = dict(options or {})
    return _option_enabled(options.get("pin_memory"), True)


def _looks_like_rlds_dataset(dataset: Any) -> bool:
    """True when ``dataset`` is (a subclass instance of) an RLDS dataset.

    The wrapped dataset classes carry a class marker; the module/name check is a
    fallback for unwrapped original ``RLDSDataset`` / ``EpisodicRLDSDataset``
    instances.  No prismatic import is triggered here.
    """
    cls = type(dataset)
    if getattr(cls, "__turbo_physai_rlds_dataset__", False):
        return True
    return (
        cls.__module__ == "prismatic.vla.datasets.datasets"
        and cls.__name__ in ("RLDSDataset", "EpisodicRLDSDataset")
    )


def _install_identity(subclass: type, original: type) -> type:
    subclass.__name__ = original.__name__
    subclass.__qualname__ = original.__qualname__
    subclass.__doc__ = original.__doc__
    return subclass


# --- dataset spawn-pickling wrappers -----------------------------------------

def _spawn_picklable_dataset(original: type) -> type:
    """Subclass ``original`` so instances can cross a spawn process boundary.

    An RLDS dataset owns a TensorFlow graph (``self.dataset``) that cannot be
    pickled.  The subclass remembers the constructor arguments and serialises
    them instead (via ``__reduce__``); unpickling inside a spawn worker runs the
    real constructor, which rebuilds the TF graph from scratch there.

    The base class is serialised as a **module/qualname path**, not as the class
    object: pickle pickles classes by reference and verifies that the module
    attribute still points at the object being pickled.  Applying this Group
    replaces ``prismatic.vla.datasets.datasets.RLDSDataset`` with the wrapped
    subclass, so handing pickle the original class object fails with
    ``"it's not the same object as ..."``.  Resolving the path at *unpickle*
    time works because the spawn worker imports prismatic fresh — the framework
    patch is not applied there, so the module attribute is the original class.
    """

    base_module = original.__module__
    base_qualname = original.__qualname__

    class _SpawnPicklableDataset(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            # Keep the construction arguments for __reduce__ (spawn pickling).
            self.__ctor_args = args
            self.__ctor_kwargs = kwargs
            super().__init__(*args, **kwargs)

        def __reduce__(self):
            # The TF graph cannot be pickled; send the constructor arguments and a
            # *path* to the base class, then rebuild the dataset from scratch inside
            # the spawn worker.
            return (
                _reconstruct_rlds_dataset,
                (base_module, base_qualname, self.__ctor_args, self.__ctor_kwargs),
            )

    _SpawnPicklableDataset.__turbo_physai_rlds_dataset__ = True
    return _install_identity(_SpawnPicklableDataset, original)


def _reconstruct_rlds_dataset(
    module_name: str, class_path: str, args: tuple, kwargs: dict
):
    """Module-level reconstructor used by ``__reduce__`` (must be importable).

    Resolves the class from ``module_name`` / ``class_path`` **in the unpickling
    process** (a spawn DataLoader worker, where prismatic is imported fresh and
    the module attribute is the unpatched original class), then runs its real
    constructor to rebuild the TF graph.
    """
    import importlib

    module = importlib.import_module(module_name)
    cls = module
    for part in class_path.split("."):
        cls = getattr(cls, part)
    cls = _seeded_dataset_class(cls)
    return cls(*args, **kwargs)


def _seeded_dataset_class(cls: type) -> type:
    """Let the reproducibility Groups give each worker its own data stream.

    This is the only place where patched behaviour can enter a spawn worker: the
    worker re-imports the model stack, so the framework's replacements are not
    installed there, while this function is reached through the pickle payload.

    The hook is inert unless ``openvla.reproducibility.*`` published a run seed in
    the parent process (its environment marker is inherited by the worker), i.e.
    this Group keeps its exact behaviour on its own.
    """
    try:
        from .reproducibility import worker_seeded_dataset_class
    except Exception:  # pragma: no cover - a broken import must not break pickling
        return cls
    return worker_seeded_dataset_class(cls)


def rlds_dataset_spawn_wrapper(
    original: type, options: Optional[Mapping[str, Any]] = None
) -> type:
    """``wrap`` factory for ``RLDSDataset`` (``(original, options) -> subclass``)."""

    del options
    return _spawn_picklable_dataset(original)


def episodic_rlds_dataset_spawn_wrapper(
    original: type, options: Optional[Mapping[str, Any]] = None
) -> type:
    """``wrap`` factory for ``EpisodicRLDSDataset`` (``(original, options) -> subclass``)."""

    del options
    return _spawn_picklable_dataset(original)


# --- DataLoader spawn/worker wrapper ------------------------------------------

def dataloader_spawn_wrapper(
    original: type, options: Optional[Mapping[str, Any]] = None
) -> type:
    """``wrap`` factory for ``torch.utils.data.DataLoader``.

    Returned subclass forces ``num_workers=N`` (Group option, default 1), the
    ``spawn`` multiprocessing context and — when ``N > 0`` — ``pin_memory=True``
    (Group option ``pin_memory``, default ``True``) **only when the dataset is an
    RLDS dataset**; all other DataLoader constructions pass through unchanged.

    The `num_workers` option lives in this Group rather than in the upstream
    DataLoader call so the item stays switchable/AB-testable from the recipe.
    """

    num_workers = _rlds_num_workers(options)
    pin_memory = _rlds_pin_memory(options)

    class _SpawnWorkerDataLoader(original):
        def __init__(self, dataset: Any, *args: Any, **kwargs: Any) -> None:
            if _looks_like_rlds_dataset(dataset):
                kwargs["num_workers"] = num_workers
                if num_workers > 0:
                    kwargs["multiprocessing_context"] = _spawn_context()
                    # Pinned host memory only pays off when a worker builds the batch
                    # ahead of the main process; with `num_workers=0` there is nothing
                    # to overlap with, so leave the baseline default (False) in place.
                    if pin_memory:
                        kwargs["pin_memory"] = True
            super().__init__(dataset, *args, **kwargs)

    return _install_identity(_SpawnWorkerDataLoader, original)


_SPAWN_CONTEXT = None


def _spawn_context():
    global _SPAWN_CONTEXT
    if _SPAWN_CONTEXT is None:
        import torch.multiprocessing as torch_mp

        _SPAWN_CONTEXT = torch_mp.get_context("spawn")
    return _SPAWN_CONTEXT
