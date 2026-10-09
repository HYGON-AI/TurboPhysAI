# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA reproducibility: a reproducible data stream and consistent kernels.

This module implements two independent Groups.  They answer two different
questions and they complement each other:

``openvla.reproducibility.data_order`` -- *which* samples does a step see?
    Upstream OpenVLA seeds ``random`` / ``numpy`` / ``torch`` in
    ``prismatic.util.torch_utils.set_global_seed`` and stops there, but the RLDS
    pipeline (``dlimp`` / ``tfds``) constructs several ops with ``seed=None``:
    ``Dataset.shuffle()`` and ``tf.data.Dataset.sample_from_datasets()`` inside
    ``make_interleaved_dataset``, TFDS' file-level ``shuffle_files=True`` (dlimp
    passes no ``shuffle_seed``) and the ``tf.random.uniform`` draw in the frame
    transform.  TensorFlow resolves ``seed=None`` from its *global* RNG when the op
    is constructed, so without seeding it the sample order is re-randomized on
    every launch and two runs cannot be compared step by step.

    The Group therefore pins the run seed before the RLDS graph is built, and
    threads it explicitly into the mixture sampling and the *frame* shuffle -- the
    two places the validated fork seeds explicitly.  TFDS' file-level shuffle is
    deliberately left to the global RNG, exactly as the fork leaves it; see
    :func:`dlimp_shuffle_wrapper` for why pinning that one too would *change* the
    batch stream instead of reproducing the fork's.

``openvla.reproducibility.determinism`` -- do the same samples *compute* the same?
    Seeds say nothing about kernel selection: (c)uDNN/MIOpen may autotune to a
    different, non-deterministic convolution, TF32 truncates the fp32 mantissa,
    and some backward reductions accumulate in a non-deterministic order.  This
    Group pins those choices, and re-asserts the run seed after ``PrismaticVLM``
    construction -- whose ``__init__`` calls
    ``torch.manual_seed(vision_backbone.embed_dim)`` and would otherwise *become*
    the effective torch seed of the run.

    It anchors on ``PrismaticVLM.__init__`` because that constructor is the last
    thing to touch the global RNG before the training loop, the first point every
    training/eval entry point has certainly reached, and the earliest point at
    which the run seed is known (upstream sets it before loading the model).  It
    also collides with no other Group: ``set_global_seed`` belongs to
    ``data_order``, ``run_vla_training`` to ``openvla.gc.freeze``, ``run_setup``
    to ``openvla.compile.fsdp1`` and the ``DataLoader`` to
    ``openvla.dataloader.spawn``.

Only both Groups together give a bit-reproducible loss.  ``data_order`` alone
fixes the samples but leaves kernel noise (the loss then agrees to a few
significant digits); ``determinism`` alone fixes the arithmetic while the samples
still differ from run to run.  Container-level determinism
(``MIOPEN_DEBUG_CONVOLUTION_DETERMINISTIC``, ``MIOPEN_FIND_MODE``, the rocBLAS
tuning library) is deliberately *not* here: those variables are read when the
backend initializes and have to be set by the RuntimeConfig before the
interpreter starts.

Where the run seed travels
--------------------------
``set_global_seed`` publishes the seed in-process and in the environment
(``TURBO_PHYSAI_OPENVLA_SEED``).  The environment copy is what makes spawn
DataLoader workers work: they re-import the model stack, so the framework's
replacements are not installed there, and the marker is inherited instead.  When
only ``data_order`` is enabled the seed is still found, through the upstream
``EXPERIMENT_GLOBAL_SEED`` marker that ``prismatic`` itself writes.  With no seed
published at all every wrapper is a pass-through, i.e. the baseline behaviour is
bit-identical.

Options
-------
``openvla.reproducibility.data_order``
    ``explicit_seeds`` (bool, default ``true``)
        Thread the run seed into ``dlimp``'s mixture sampling and into the
        ``tf.data`` frame shuffle, on top of the global TF seed.
    ``hold_shuffle_permutation`` (bool, default ``true``)
        Pass ``reshuffle_each_iteration=False`` on the seeded frame shuffle, so
        the permutation is a function of the seed alone.

``openvla.reproducibility.determinism``
    The five options mirror the torch attributes they set; their defaults are the
    bundle the validated fork applies behind ``OPENVLA_DETERMINISTIC=1``:
    ``cudnn_deterministic`` (true), ``benchmark`` (false), ``allow_tf32`` (false),
    ``deterministic_algorithms`` (true) and ``strict`` (false).  ``strict`` makes
    ``use_deterministic_algorithms`` hard-fail instead of warn on the few ops
    that have no deterministic kernel on ROCm.

Known limits
------------
* ``--image_aug`` is not reproducible: the augmentation seed is drawn inside a
  ``tf.data`` map running with 16 parallel calls, so the values a frame receives
  depend on the order in which the replicas consume the stateful op.
* Bit-for-bit equality is a same-machine, same-world-size property.  Different
  device counts shard and reduce differently, and the RLDS normalization
  statistics can be recomputed with a different file read order.
* ``metrics.jsonl`` is written per rank and never reduced across ranks, so a
  comparison is between the same rank of two runs.

Module import is side-effect free and does not import torch / tensorflow /
prismatic: every dependency is imported inside the function that needs it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from typing import Any, Callable, Optional

__all__ = [
    "RUN_SEED_ENV",
    "active_run_seed",
    "apply_determinism",
    "dlimp_shuffle_wrapper",
    "make_dataset_from_rlds_wrapper",
    "make_interleaved_dataset_wrapper",
    "prismatic_vlm_init_wrapper",
    "reseed_host_rngs",
    "run_seed",
    "sample_from_datasets_wrapper",
    "seed_tensorflow",
    "set_global_seed_wrapper",
    "set_run_seed",
    "worker_init_function_wrapper",
    "worker_seeded_dataset_class",
]


#: Seed marker of these Groups; unlike a process global it survives ``spawn``.
RUN_SEED_ENV = "TURBO_PHYSAI_OPENVLA_SEED"
#: Seed marker upstream ``prismatic`` writes in its own ``set_global_seed``.
UPSTREAM_SEED_ENV = "EXPERIMENT_GLOBAL_SEED"

_RUN_SEED: Optional[int] = None

#: The wrapped ``worker_init_function`` built by this module's factory.
#: ``set_global_seed`` hands it to the DataLoader, so the two members of the
#: ``data_order`` Group have to agree on one object.
_WORKER_INIT: Optional[Callable[[int], None]] = None


# --- options and the run seed ------------------------------------------------

def _option(options: Optional[Mapping[str, Any]], name: str, default: Any) -> Any:
    """Read one Group option, with a default for a missing or empty mapping."""

    return dict(options or {}).get(name, default)


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


def _env_seed(name: str) -> Optional[int]:
    """Read an integer seed from the environment; ``None`` when absent or invalid."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def set_run_seed(seed: int) -> int:
    """Record the run seed in-process and publish it to spawned children.

    The environment copy is not a convenience: a spawn DataLoader worker
    re-imports the model stack and therefore never sees this module's globals.
    """

    global _RUN_SEED
    _RUN_SEED = int(seed)
    os.environ[RUN_SEED_ENV] = str(_RUN_SEED)
    return _RUN_SEED


def run_seed() -> Optional[int]:
    """Effective run seed, or ``None`` when nothing has been seeded yet.

    Resolution order: the in-process record (written by this Group's
    ``set_global_seed`` wrapper), then this Group's environment marker, then the
    upstream ``EXPERIMENT_GLOBAL_SEED`` marker that ``prismatic`` itself writes.
    The last step is what lets ``data_order`` work with the ``seed`` half
    disabled and what lets ``determinism`` re-seed a run that only seeded the
    host libraries.
    """

    if _RUN_SEED is not None:
        return _RUN_SEED
    for name in (RUN_SEED_ENV, UPSTREAM_SEED_ENV):
        seed = _env_seed(name)
        if seed is not None:
            return seed
    return None


def active_run_seed() -> Optional[int]:
    """The seed *these* Groups published, i.e. the environment marker only.

    Used where the answer has to mean "the framework published a seed in the
    parent process" -- notably :func:`worker_seeded_dataset_class`, which runs
    inside a spawn worker and must stay inert when only
    ``openvla.dataloader.spawn`` is enabled.
    """

    return _env_seed(RUN_SEED_ENV)


# --- library-level seeding ---------------------------------------------------

def _tensorflow():
    """Return the TensorFlow module, or ``None`` when it is not installed."""

    try:
        import tensorflow
    except ImportError:
        return None
    return tensorflow


def seed_tensorflow(seed: int) -> bool:
    """Pin TensorFlow's *global* RNG; returns whether TF was available.

    Has to run before any TF op is constructed for the seed to matter, because TF
    resolves ``seed=None`` from this global state at graph-construction time.
    """

    tensorflow = _tensorflow()
    if tensorflow is None:
        return False
    tensorflow.random.set_seed(int(seed))
    return True


def apply_determinism(
    cudnn_deterministic: bool = True,
    benchmark: bool = False,
    allow_tf32: bool = False,
    deterministic_algorithms: bool = True,
    strict: bool = False,
) -> bool:
    """Pin kernel selection on top of the seeds; returns whether torch was available.

    Every argument mirrors the torch attribute it sets:

    * ``cudnn_deterministic`` / ``benchmark``: deterministic kernels only, and no
      runtime autotuning (on ROCm this is the MIOpen path).  Autotuning is the
      usual reason two identical runs pick different algorithms.
    * ``allow_tf32`` for matmul and (c)udnn: TF32 truncates the fp32 mantissa, so
      it has to be off for a run that is compared against another vendor.
    * ``deterministic_algorithms`` with ``strict=False``: deterministic kernels
      and reductions, warning instead of aborting on the handful of ops ROCm has
      no deterministic implementation for; ``strict=True`` hard-fails instead.
      When disabled, the process setting is left untouched.
    """

    try:
        import torch
    except ImportError:
        return False

    torch.backends.cudnn.deterministic = bool(cudnn_deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    except (AttributeError, RuntimeError):  # backend without a CUDA/TF32 config
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    except (AttributeError, RuntimeError):  # backend without a (c)udnn TF32 config
        pass
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=not strict)
    return True


def reseed_host_rngs(seed: Optional[int]) -> bool:
    """Make ``seed`` the effective torch seed again; returns whether it ran.

    ``torch.manual_seed`` already forwards to ``torch.cuda.manual_seed_all``, but
    the flash-attention dropout path and any per-device ``torch.Generator`` read
    the per-device generators directly, so the call is made explicit.
    """

    if seed is None:
        return False
    try:
        import torch
    except ImportError:
        return False
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return True


# --- Group ``openvla.reproducibility.data_order`` -----------------------------

def set_global_seed_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``prismatic.util.torch_utils.set_global_seed``.

    Keeps the baseline seeding of ``random`` / ``numpy`` / ``torch`` exactly as it
    is and adds the two data-stream pieces on top:

    * the run seed is recorded and published, which is what the RLDS pipeline
      wrappers below and the spawn DataLoader workers read;
    * TensorFlow's global RNG is seeded, which is what pins the ``seed=None`` ops
      of the RLDS pipeline (see the module docstring).

    The returned ``worker_init_fn`` is this module's wrapped version, so workers
    get a TensorFlow child sequence as well.
    """

    del options

    @functools.wraps(original)
    def set_global_seed(seed: int, get_worker_init_fn: bool = False):
        original(seed, get_worker_init_fn)  # random / numpy / torch + EXPERIMENT_GLOBAL_SEED
        set_run_seed(seed)
        seed_tensorflow(seed)
        return _WORKER_INIT if get_worker_init_fn else None

    return set_global_seed


def _worker_tensorflow_seed(worker_id: int) -> Optional[int]:
    """TensorFlow child seed for one DataLoader worker.

    Mirrors the derivation the baseline ``worker_init_function`` already uses and
    extends it with a third child sequence for TensorFlow::

        seed_seq = SeedSequence([base_seed, worker_id, LOCAL_RANK])
        tf_seed  = seed_seq.spawn(3)[2]

    ``SeedSequence.spawn(n)`` derives child ``i`` from ``(spawn_key + (i,))``
    alone, so asking for three children instead of two leaves the torch and
    ``random`` seeds bit-identical to the baseline.

    Has to be called *before* the baseline ``worker_init_function``: that one
    reseeds torch, after which ``torch.initial_seed()`` no longer holds the
    per-worker seed the DataLoader installed.
    """

    try:
        import numpy as np
        import torch
    except ImportError:
        return None
    global_rank = int(os.environ.get("LOCAL_RANK", 0))
    base_seed = int(torch.initial_seed()) - int(worker_id)
    seed_sequence = np.random.SeedSequence([base_seed, int(worker_id), global_rank])
    return int(seed_sequence.spawn(3)[2].generate_state(1, dtype=np.uint64)[0])


def worker_init_function_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``prismatic.util.torch_utils.worker_init_function``.

    Adds a TensorFlow child seed on top of the baseline worker seeding.  This
    runs *after* a spawned worker has unpickled its dataset, so it cannot
    influence a TF graph that was already built: the per-worker *data stream*
    seed is applied by :func:`worker_seeded_dataset_class` instead.
    """

    del options

    @functools.wraps(original)
    def worker_init_function(worker_id: int) -> None:
        tf_seed = _worker_tensorflow_seed(worker_id)
        original(worker_id)
        if tf_seed is not None:
            seed_tensorflow(tf_seed)

    global _WORKER_INIT
    _WORKER_INIT = worker_init_function
    return worker_init_function


def _seed_before_pipeline_build(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """Shared factory for the two RLDS pipeline builders.

    TensorFlow captures ``seed=None`` for every dataset op (TFDS' file shuffle,
    ``sample_from_datasets``, the frame ``shuffle`` and the augmentation draws) at
    graph-construction time, so the run seed has to be in the global RNG *before*
    the wrapped call builds those ops -- in every process that builds the graph,
    including spawn DataLoader workers.

    A ``seed`` keyword supplied by the caller (the validated fork threads one)
    wins over the run seed and is consumed here instead of being forwarded,
    because upstream's signature has no such parameter.
    """

    del options

    @functools.wraps(original)
    def builder(*args: Any, **kwargs: Any) -> Any:
        seed = kwargs.pop("seed", None)
        effective_seed = seed if seed is not None else run_seed()
        if effective_seed is not None:
            seed_tensorflow(effective_seed)
        return original(*args, **kwargs)

    return builder


def make_dataset_from_rlds_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``make_dataset_from_rlds``.

    ``dlimp.DLataset.from_rlds(..., shuffle=True)`` shuffles the *files* through
    TFDS, which resolves its seed from TensorFlow's global RNG, and the rest of
    that dataset's pipeline is constructed below this call -- so this is the
    earliest point of the RLDS build and the right place to pin the seed.
    """

    return _seed_before_pipeline_build(original, options)


def make_interleaved_dataset_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``make_interleaved_dataset``.

    The same backstop as :func:`make_dataset_from_rlds_wrapper`, applied to the
    whole mixture build: per-dataset file order, the frame-level mixture sampling
    and the frame shuffle all come out of the TF ops constructed below this call.
    """

    return _seed_before_pipeline_build(original, options)


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    """Whether ``function`` accepts a keyword argument ``name``."""

    try:
        import inspect

        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def sample_from_datasets_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``dlimp.DLataset.sample_from_datasets``.

    ``make_interleaved_dataset`` calls this without a seed and dlimp forwards
    ``seed=None`` to ``tf.data.Dataset.sample_from_datasets`` -- a fresh choice
    stream per launch, which decides *which* dataset every frame is drawn from.
    With ``explicit_seeds`` (default on) a missing seed becomes the run seed; a
    seed the caller passed is never touched, and a dlimp without a ``seed``
    parameter is still called as before instead of raising ``TypeError``.
    """

    explicit_seeds = _option_enabled(_option(options, "explicit_seeds", True), True)
    supports_seed = _accepts_keyword(original, "seed")

    @functools.wraps(original)
    def sample_from_datasets(
        datasets: Any, weights: Any = None, seed: Optional[int] = None, **kwargs: Any
    ) -> Any:
        if explicit_seeds and seed is None:
            seed = run_seed()
        if supports_seed:
            kwargs["seed"] = seed
        return original(datasets, weights, **kwargs)

    return sample_from_datasets


def dlimp_shuffle_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``dlimp.DLataset.shuffle`` -- the mixture's *frame* shuffle.

    The frame shuffle is written ``dataset.shuffle(shuffle_buffer_size)`` inside
    ``make_interleaved_dataset``.  With ``explicit_seeds`` (default on) a missing seed
    becomes the run seed, and with ``hold_shuffle_permutation`` (default on) the
    permutation is fixed to that seed instead of being redrawn per iteration.  Only
    ``seed=None`` calls are touched: an explicit seed from any other caller is
    preserved, and with no run seed published the wrapper is a pass-through.

    Why the anchor is ``dlimp.DLataset`` and not ``tf.data.Dataset.shuffle``
    ----------------------------------------------------------------------
    ``tf.data.Dataset.shuffle`` is also the method TFDS uses for the *file-level*
    shuffle of ``from_rlds(..., shuffle=True)``
    (``instruction_ds.shuffle(len(files), seed=read_config.shuffle_seed)``, with
    dlimp passing no ``shuffle_seed``).  Patching that method rewrites the shard
    order as well, and the validated fork deliberately leaves *that* one to the
    global RNG: under eager execution ``seed=None`` resolves to
    ``(global_seed, random.Random(global_seed).randint(0, 2**31 - 1))``, i.e. it
    depends only on the global seed and on how many seedless random ops were built
    since it was set -- never on the op count of the process.  Both spellings are
    reproducible, but they read the shards in *different* orders, so pinning the
    file shuffle here would produce a batch stream that no longer matches the
    fork's (first batch included).  Anchoring on ``DLataset`` hits only the frame
    shuffle that ``make_interleaved_dataset`` performs, which is the one the fork
    seeds explicitly.  ``DLataset`` reaches this wrapper like any other
    ``tf.data.Dataset`` method: its ``__getattribute__`` re-wraps the result into a
    ``DLataset``.
    """

    explicit_seeds = _option_enabled(_option(options, "explicit_seeds", True), True)
    hold_permutation = _option_enabled(_option(options, "hold_shuffle_permutation", True), True)

    @functools.wraps(original)
    def shuffle(
        self: Any,
        buffer_size: Any,
        seed: Optional[int] = None,
        reshuffle_each_iteration: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        if explicit_seeds and seed is None:
            effective_seed = run_seed()
            if effective_seed is not None:
                seed = int(effective_seed)
                if hold_permutation and reshuffle_each_iteration is None:
                    reshuffle_each_iteration = False
        return original(
            self,
            buffer_size,
            seed=seed,
            reshuffle_each_iteration=reshuffle_each_iteration,
            **kwargs,
        )

    return shuffle


# --- Group ``openvla.reproducibility.determinism`` ----------------------------

def prismatic_vlm_init_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """``wrap`` factory for ``prismatic.models.vlms.prismatic.PrismaticVLM.__init__``.

    ``PrismaticVLM.__init__`` calls ``torch.manual_seed(vision_backbone.embed_dim)``
    so that the projector initialization is deterministic -- which *overwrites*
    the run seed, and would otherwise make the ViT embedding dim the effective
    torch seed of the run.  Re-asserting the run seed here is what makes
    ``--seed`` mean what it says, and doing it in the constructor (rather than
    after model loading in the training script) also covers evaluation scripts
    and any other entry point that builds a VLM.

    Nothing between the construction and the re-seed consumes randomness, and
    nothing before the construction runs a real kernel, so neither the
    determinism bundle nor the re-seed moves a draw.
    """

    settings = {
        "cudnn_deterministic": _option_enabled(_option(options, "cudnn_deterministic", True), True),
        "benchmark": _option_enabled(_option(options, "benchmark", False), False),
        "allow_tf32": _option_enabled(_option(options, "allow_tf32", False), False),
        "deterministic_algorithms": _option_enabled(
            _option(options, "deterministic_algorithms", True), True
        ),
        "strict": _option_enabled(_option(options, "strict", False), False),
    }

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        apply_determinism(**settings)
        original(self, *args, **kwargs)
        reseed_host_rngs(run_seed())

    return __init__


# --- Per-worker data streams (used by ``openvla.dataloader.spawn``) -----------

def _worker_id() -> int:
    """DataLoader worker id, or ``0`` outside a worker (``spawn`` unpickling)."""

    try:
        import torch.utils.data
    except ImportError:
        return 0
    info = torch.utils.data.get_worker_info()
    return 0 if info is None else int(info.id)


def _install_identity(subclass: type, original: type) -> type:
    """Give ``subclass`` the original class' name/doc, like the spawn wrapper does."""

    subclass.__name__ = original.__name__
    subclass.__qualname__ = original.__qualname__
    subclass.__doc__ = original.__doc__
    return subclass


def worker_seeded_dataset_class(original: type, seed: Optional[int] = None) -> type:
    """Return an RLDS dataset class whose TF graph is built per DataLoader worker.

    Called by the spawn rebuild path (``spawn_dataloader._reconstruct_rlds_dataset``)
    inside a worker: the worker re-runs the dataset constructor there, which
    happens *before* ``worker_init_fn`` and therefore too early for anything
    worker-id dependent, and (without this hook) against an unseeded global RNG --
    i.e. a different data stream per worker and per launch.

    The returned subclass applies the validated fork's rule, ``run_seed + worker_id``:

    * its constructor seeds TensorFlow before ``super().__init__`` builds the
      graph, and remembers the seed the live graph was built with;
    * ``__iter__`` is where the worker id becomes known -- the DataLoader fetcher
      calls ``iter(dataset)`` after ``worker_init_fn`` -- so the graph is rebuilt
      there whenever the remembered seed is not this worker's;
    * two co-existing workers each iterate a private copy of the same infinite
      stream, so the ``+ worker_id`` offset is what keeps the streams disjoint:
      with one shared seed every batch would be emitted twice.

    ``original`` is returned unchanged when no seed was published by these Groups
    (so ``openvla.dataloader.spawn`` keeps its exact behaviour on its own) and when
    the class has already been wrapped.
    """

    base_seed = active_run_seed() if seed is None else int(seed)
    if base_seed is None or getattr(original, "__turbo_physai_worker_seeded__", False):
        return original

    class _WorkerSeededDataset(original):
        """RLDS dataset whose TensorFlow graph is rebuilt per DataLoader worker."""

        __turbo_physai_rlds_dataset__ = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.__ctor_args = args
            self.__ctor_kwargs = kwargs
            self._turbo_physai_seed_graph()
            super().__init__(*args, **kwargs)

        def _turbo_physai_seed_graph(self) -> int:
            """Seed TensorFlow for this worker and remember the live graph's seed."""

            wanted = base_seed + _worker_id()
            seed_tensorflow(wanted)
            self.__built_seed = wanted
            return wanted

        def __iter__(self):
            if self.__built_seed != base_seed + _worker_id():
                # First `iter()` inside a worker: rebuild the graph for this
                # worker's seed (worker 0 reuses the graph its constructor built).
                self._turbo_physai_seed_graph()
                original.__init__(self, *self.__ctor_args, **self.__ctor_kwargs)
            # Delegate to the base implementation instead of iterating here, so
            # subclasses that yield differently (`EpisodicRLDSDataset`) keep their
            # exact semantics.
            return super().__iter__()

    _WorkerSeededDataset.__turbo_physai_worker_seeded__ = True
    return _install_identity(_WorkerSeededDataset, original)
