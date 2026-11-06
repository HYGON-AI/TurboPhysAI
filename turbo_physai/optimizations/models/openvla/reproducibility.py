# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA 可复现性：数据流可复现 + kernel 选择固定。

两个独立 Group：`data_order` 决定「第 k 步看到哪些样本」，`determinism` 决定
「同样的样本是否算出同样的数」。

为什么要做：上游只固定 random/numpy/torch的随机种子，
而 dlimp/TFDS的 shuffle、sample_from_datasets、frame shuffle 都以 seed=None 构造，
TF 在构图时从全局 RNG 取值，导致每次启动的样本顺序都不同。
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


# --- fixed behaviour and the run seed ----------------------------------------

#: ``data_order``: 缺失的 seed 一律回落到 run seed，frame shuffle 固定置换。
EXPLICIT_SEEDS = True
HOLD_SHUFFLE_PERMUTATION = True

#: ``determinism``: 固定的 torch 后端设置。
DETERMINISM = {
    "cudnn_deterministic": True,
    "benchmark": False,
    "allow_tf32": False,
    "deterministic_algorithms": True,
    "strict": False,
}


def _env_seed(name: str) -> Optional[int]:
    """从环境变量读整数 seed；缺失或非法返回 None。"""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def set_run_seed(seed: int) -> int:
    """记录 run seed 并发布到环境变量。

    spawn worker 看不到本模块的全局变量，只能读环境。
    """

    global _RUN_SEED
    _RUN_SEED = int(seed)
    os.environ[RUN_SEED_ENV] = str(_RUN_SEED)
    return _RUN_SEED


def run_seed() -> Optional[int]:
    """当前生效的 run seed。

    解析顺序：进程内记录 → 本组环境标记 → 上游 EXPERIMENT_GLOBAL_SEED。
    """

    if _RUN_SEED is not None:
        return _RUN_SEED
    for name in (RUN_SEED_ENV, UPSTREAM_SEED_ENV):
        seed = _env_seed(name)
        if seed is not None:
            return seed
    return None


def active_run_seed() -> Optional[int]:
    """只认本组发布的环境标记。

    spawn worker 用它判断父进程是否播种；spawn 单独开时保持惰性。
    """

    return _env_seed(RUN_SEED_ENV)


# --- library-level seeding ---------------------------------------------------

def _tensorflow():
    """取 tensorflow 模块；没装返回 None。"""

    try:
        import tensorflow
    except ImportError:
        return None
    return tensorflow


def seed_tensorflow(seed: int) -> bool:
    """固定 TF 全局 RNG。

    必须在任何 TF op 构造前调用：TF 构图时从全局 RNG 解析 seed=None。
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
    """固定 kernel 选择：关掉 autotune 和 TF32，打开 deterministic_algorithms。

    strict=False，ROCm 上没有确定性实现的算子只警告不中断。
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
    """重新把 seed 设为 torch 的有效种子。

    torch.manual_seed 已转发 cuda，但 FA dropout 和 per-device Generator 直接读
    per-device 生成器，所以显式再调一次。
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
    """保留 baseline 的 random/numpy/torch 播种，额外做两件事。

    记录并发布 run seed、播 TF 全局 RNG，并返回本模块包装过的 worker_init_fn。
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
    """给一个 worker 派生 TF 子种子。

    沿用 baseline 的 SeedSequence，多 spawn 一个子序列给 TF，torch/random 的种子与
    baseline 逐位一致。必须在 baseline worker_init_function 之前调用，否则
    initial_seed() 已被改写。
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
    """在 baseline 播种之上补 TF 子种子。它跑在 worker 已
    unpickle dataset 之后，改不了已建好的图；per-worker 数据流由
    worker_seeded_dataset_class 负责。
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
    """两个 RLDS 建图函数的共用工厂。

    TF 在构图时捕获 seed=None，所以要在 wrapped 调用建图前把 run seed 放进全局 RNG。
    调用方显式传的 seed 优先并被使用（上游签名没有这个参数）。
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
    return _seed_before_pipeline_build(original, options)


def make_interleaved_dataset_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """同上，覆盖整个 mixture 构建：文件顺序、mixture 采样和 frame shuffle 都在此调用下构造。
    """

    return _seed_before_pipeline_build(original, options)


def _accepts_keyword(function: Callable[..., Any], name: str) -> bool:
    """function 是否接受关键字参数 name。"""

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
    """dlimp 的 mixture 采样：缺失的 seed 用 run seed。"""

    del options

    supports_seed = _accepts_keyword(original, "seed")

    @functools.wraps(original)
    def sample_from_datasets(
        datasets: Any, weights: Any = None, seed: Optional[int] = None, **kwargs: Any
    ) -> Any:
        if EXPLICIT_SEEDS and seed is None:
            seed = run_seed()
        if supports_seed:
            kwargs["seed"] = seed
        return original(datasets, weights, **kwargs)

    return sample_from_datasets


def dlimp_shuffle_wrapper(
    original: Callable[..., Any], options: Optional[Mapping[str, Any]] = None
) -> Callable[..., Any]:
    """
    mixture 的 frame shuffle：缺失 seed 用 run seed，并把置换固定下来（reshuffle_each_iteration=False）。
    """

    del options

    @functools.wraps(original)
    def shuffle(
        self: Any,
        buffer_size: Any,
        seed: Optional[int] = None,
        reshuffle_each_iteration: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        if EXPLICIT_SEEDS and seed is None:
            effective_seed = run_seed()
            if effective_seed is not None:
                seed = int(effective_seed)
                if HOLD_SHUFFLE_PERMUTATION and reshuffle_each_iteration is None:
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
    """
    PrismaticVLM.__init__ 会覆盖 run seed，这里重新落实。
    """

    del options

    @functools.wraps(original)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        apply_determinism(**DETERMINISM)
        original(self, *args, **kwargs)
        reseed_host_rngs(run_seed())

    return __init__


# --- Per-worker data streams (used by ``openvla.dataloader.spawn``) -----------

def _worker_id() -> int:
    """DataLoader worker id；非 worker（如 spawn unpickle 阶段）返回 0。"""

    try:
        import torch.utils.data
    except ImportError:
        return 0
    info = torch.utils.data.get_worker_info()
    return 0 if info is None else int(info.id)


def _install_identity(subclass: type, original: type) -> type:
    """让子类沿用原类的名字和 doc。"""

    subclass.__name__ = original.__name__
    subclass.__qualname__ = original.__qualname__
    subclass.__doc__ = original.__doc__
    return subclass


def worker_seeded_dataset_class(original: type, seed: Optional[int] = None) -> type:
    """
    按 run_seed + worker_id 播种的 RLDS 数据集子类。
    """

    base_seed = active_run_seed() if seed is None else int(seed)
    if base_seed is None or getattr(original, "__turbo_physai_worker_seeded__", False):
        return original

    class _WorkerSeededDataset(original):
        """按 DataLoader worker 重建 TF 图的 RLDS 数据集。"""

        __turbo_physai_rlds_dataset__ = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.__ctor_args = args
            self.__ctor_kwargs = kwargs
            self._turbo_physai_seed_graph()
            super().__init__(*args, **kwargs)

        def _turbo_physai_seed_graph(self) -> int:
            """播种当前 worker 的 TF，并记下这次建图用的 seed。"""

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
