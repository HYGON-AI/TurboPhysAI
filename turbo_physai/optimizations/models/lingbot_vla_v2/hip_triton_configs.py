# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""inductor 点式/规约 kernel 的 autotune 候选集补充（HCU/HIP）。"""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Mapping, Optional, Sequence

log = logging.getLogger(__name__)

DEBUG = os.environ.get("TURBOPHYSAI_HIP_TRITON_DEBUG") == "1"

POINTWISE_1D: Sequence[tuple] = (
    (8192, 8, None, 1),
    (8192, 2, None, 1),
    (8192, 8, 1, 1),
    (4096, 8, None, 1),
    (4096, 2, None, 1),
    (4096, 8, 1, 1),
    (2048, 8, None, 1),
    (1024, 8, None, 1),
)

POINTWISE_2D: Sequence[tuple] = (
    (128, 64, 8),
    (64, 128, 8),
    (256, 32, 8),
    (32, 256, 8),
    (512, 32, 8),
)

REDUCTION: Sequence[tuple] = (
    (16, 32, 4),
    (16, 64, 4),
    (32, 32, 4),
    (16, 256, 4),
    (32, 128, 4),
    (64, 64, 4),
    (128, 32, 8),
    (32, 256, 8),
)


def _tables(options: Optional[Mapping[str, Any]]) -> tuple:

    def pick(key: str, default: Sequence[tuple]) -> Sequence[tuple]:
        value = (options or {}).get(key)
        if not value:
            return default
        return tuple(tuple(item) for item in value)

    return (
        pick("pointwise_1d", POINTWISE_1D),
        pick("pointwise_2d", POINTWISE_2D),
        pick("reduction", REDUCTION),
    )


def _make_config(th, kwargs: Mapping[str, Any], num_warps: int, num_stages: int = 1):

    cfg = th.Config(dict(kwargs), num_warps=num_warps, num_stages=num_stages)
    wpe = kwargs.get("waves_per_eu")
    if wpe is not None:
        cfg.kwargs["waves_per_eu"] = wpe
    return cfg


def _compatible(config, kernel_args) -> bool:
    if not kernel_args:
        return True
    return all(
        name in kernel_args
        for name in config.kwargs
        if name.endswith("BLOCK")
    )


def _pointwise_extra(th, size_hints, table_1d, table_2d):
    if not size_hints:
        return []
    keys = set(size_hints)
    out = []

    if keys == {"x"}:
        xnumel = size_hints["x"]
        for xblock, warps, wpe, stages in table_1d:
            if xnumel < xblock:
                continue
            block = {"XBLOCK": xblock}
            if wpe is not None:
                block["waves_per_eu"] = wpe
            out.append(_make_config(th, block, warps, stages))
    elif keys == {"x", "y"}:
        xnumel, ynumel = size_hints["x"], size_hints["y"]
        for xblock, yblock, warps in table_2d:
            if xnumel < xblock or ynumel < yblock:
                continue
            out.append(_make_config(th, {"XBLOCK": xblock, "YBLOCK": yblock}, warps))

    return out


def _dedupe(extra, existing):

    def key(config):
        return (
            tuple(sorted((k, v) for k, v in config.kwargs.items())),
            config.num_warps,
            config.num_stages,
        )

    seen = {key(c) for c in (existing or ())}
    out = []
    for config in extra:
        k = key(config)
        if k in seen:
            continue
        seen.add(k)
        out.append(config)
    return out


def _reduction_extra(th, size_hints, inductor_meta, table):
    if not size_hints or "x" not in size_hints:
        return []
    if not inductor_meta.get("num_reduction"):
        return []
    out = []
    for xblock, rblock, warps in table:
        if size_hints["x"] < xblock:
            continue
        try:
            cfg = th.triton_config_reduction(
                size_hints,
                int(xblock),
                int(rblock),
                num_warps=int(warps),
                reduction_hint=inductor_meta.get("reduction_hint"),
                register_intensive=False,
            )
        except Exception:
            continue
        out.append(cfg)
    return out


def pointwise(original, options: Optional[Mapping[str, Any]] = None):
    table_1d, table_2d, _ = _tables(options)

    def wrapped(*args, **kwargs):
        decorator = original(*args, **kwargs)
        size_hints = kwargs.get("size_hints")
        if size_hints is None and args:
            size_hints = args[0]

        @functools.wraps(decorator)
        def apply(fn):
            tuner = decorator(fn)
            try:
                import torch

                if not torch.version.hip:
                    return tuner
                from torch._inductor.runtime import triton_heuristics as th

                extra = _pointwise_extra(th, size_hints, table_1d, table_2d)
                kernel_args = set(getattr(fn, "arg_names", ()) or ())
                extra = [c for c in extra if _compatible(c, kernel_args)]
                extra = _dedupe(extra, tuner.configs)
                if extra:
                    before = len(list(tuner.configs or ()))
                    tuner.configs = extra + list(tuner.configs or ())
                    if DEBUG:
                        log.warning(
                            "[hip_triton_configs] %s +%d → %d configs (原 %d), "
                            "size_hints=%s",
                            getattr(fn, "__name__", "?"),
                            len(extra),
                            len(tuner.configs),
                            before,
                            size_hints,
                        )
            except Exception as exc:
                if DEBUG:
                    log.warning("[hip_triton_configs] pointwise 跳过: %s", exc)
            return tuner

        return apply

    return wrapped


def reduction_configs(original, options: Optional[Mapping[str, Any]] = None):
    _, _, table = _tables(options)

    def wrapped(*args, **kwargs):
        configs = original(*args, **kwargs)
        try:
            import torch

            if not torch.version.hip:
                return configs
            from torch._inductor.runtime import triton_heuristics as th

            extra = _reduction_extra(
                th,
                kwargs.get("size_hints"),
                kwargs.get("inductor_meta") or {},
                table,
            )
            extra = _dedupe(extra, configs)
            if extra:
                configs = extra + list(configs)
                if DEBUG:
                    log.warning(
                        "[hip_triton_configs] reduction +%d → %d configs, size_hints=%s",
                        len(extra),
                        len(configs),
                        kwargs.get("size_hints"),
                    )
        except Exception as exc:
            if DEBUG:
                log.warning("[hip_triton_configs] reduction 跳过: %s", exc)
        return configs

    return wrapped
