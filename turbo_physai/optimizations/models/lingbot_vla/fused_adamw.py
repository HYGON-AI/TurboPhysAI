# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""
开启torch原生 Fused AdamW，并让更新与梯度裁剪直接落在 FSDP2 的本地分片，数值上与原实现一致
"""

from __future__ import annotations

import functools
import logging
import types
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

try:
    from torch.distributed.tensor import DTensor
except Exception:
    DTensor = None


def _is_dtensor(value) -> bool:
    return DTensor is not None and isinstance(value, DTensor)



_LOCAL_SHARDS: dict = {}

_LOCAL_OF: dict = {}

_GLOBAL_OF: dict = {}

_SYNCED = {"flag": False}

_COMM_GROUP = {"group": None}


def _sync_grads() -> None:
    for dt, lp in _LOCAL_SHARDS.values():
        grad = dt.grad
        if grad is None:
            lp.grad = None
            continue
        local = getattr(grad, "_local_tensor", None) if _is_dtensor(grad) else grad
        if local is None:
            local = grad.to_local()
        if hasattr(local, "wait"):
            local = local.wait()
        lp.grad = local
        dt.grad = None
    _SYNCED["flag"] = True


def _localize(optim):
    """把 optimizer 的 param_groups 换成 DTensor 本地分片。"""
    mapping = {}
    for group in optim.param_groups:
        params = []
        for p in group["params"]:
            if _is_dtensor(p):
                lp = p.to_local()
                mapping[id(p)] = (p, lp)
                _LOCAL_SHARDS[id(p)] = (p, lp)
                _LOCAL_OF[id(p)] = lp
                _GLOBAL_OF[id(lp)] = p
                params.append(lp)
                try:
                    _COMM_GROUP["group"] = p.device_mesh.get_group()
                except Exception:
                    pass
            else:
                params.append(p)
        group["params"] = params

    if not mapping:
        return optim

    orig_step = optim.step

    @functools.wraps(orig_step)
    def step(self_, *args, **kwargs):
        if not _SYNCED["flag"]:
            _sync_grads()
        _SYNCED["flag"] = False
        return orig_step(*args, **kwargs)

    optim.step = types.MethodType(step, optim)
    logger.info(
        "lingbot-vla: fused AdamW on %d local shards (DTensor dispatch bypassed)",
        len(mapping),
    )
    return optim


def wrap_build_optimizer(original, _options=None):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        kwargs["fused"] = True
        optim = original(*args, **kwargs)
        return _localize(optim)

    return wrapped


@contextmanager
def _dtensor_form(optim):
    """
    checkpoint 存取时把 optimizer 临时换回 DTensor 形态。
    """
    for group in optim.param_groups:
        group["params"] = [_GLOBAL_OF.get(id(p), p) for p in group["params"]]
    state = optim.state
    optim.state = defaultdict(dict)
    for p, entry in state.items():
        dt = _GLOBAL_OF.get(id(p), p)
        local = _LOCAL_OF.get(id(dt))
        optim.state[dt] = {
            k: DTensor.from_local(v, dt.device_mesh, dt.placements, shape=dt.shape, stride=dt.stride())
            if local is not None and torch.is_tensor(v) and v.shape == local.shape
            else v
            for k, v in entry.items()
        }
    try:
        yield
    finally:
        _SYNCED["flag"] = False
        for group in optim.param_groups:
            group["params"] = [_LOCAL_OF.get(id(p), p) for p in group["params"]]
        state = optim.state
        optim.state = defaultdict(dict)
        for p, entry in state.items():
            optim.state[_LOCAL_OF.get(id(p), p)] = {
                k: v.to_local() if _is_dtensor(v) else v for k, v in entry.items()
            }


def wrap_optimizer_state_dict(original, _options=None):
    @functools.wraps(original)
    def wrapped(self):
        with _dtensor_form(self.optimizer):
            return original(self)

    return wrapped


def wrap_load_optimizer_state_dict(original, _options=None):
    @functools.wraps(original)
    def wrapped(self, state_dict):
        with _dtensor_form(self.optimizer):
            out = original(self, state_dict)
        repinned = False
        for group in self.optimizer.param_groups:
            if group.get("fused") is not True or group.get("foreach") is not False:
                repinned = True
            group["fused"] = True
            group["foreach"] = False
        if repinned:
            logger.info(
                "lingbot-vla: optimizer state 里的 fused/foreach 已覆盖，重新钉住为 fused AdamW"
            )
        return out

    return wrapped


def _all_reduce(value: torch.Tensor, op) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    group = _COMM_GROUP.get("group")
    if group is not None:
        dist.all_reduce(value, op=op, group=group)
    else:
        dist.all_reduce(value, op=op)


def _global_norm(grads, norm_type: float):
    """
    在本地分片上算 p-范数，再 all_reduce 成全局范数，口径与 torch.clip_grad_norm_ 一致。
    inf 范数是 max，需要 MAX 归约；有限 p 范数是 (sum |g|^p)^(1/p)，需要 SUM 归约。
    """
    if not grads:
        return None
    if norm_type == float("inf"):
        acc = torch.stack(list(torch._foreach_norm(grads, float("inf")))).max()
        _all_reduce(acc, dist.ReduceOp.MAX)
        return acc
    norms = torch.stack(list(torch._foreach_norm(grads, norm_type)))
    if norm_type == 1.0:
        acc = norms.sum()
    else:
        acc = (norms * norms).sum() if norm_type == 2.0 else norms.pow(norm_type).sum()
    _all_reduce(acc, dist.ReduceOp.SUM)
    if norm_type == 2.0:
        return acc.sqrt()
    if norm_type == 1.0:
        return acc
    return acc ** (1.0 / norm_type)


def wrap_clip_grad_norm(original, _options=None):
    @functools.wraps(original)
    def wrapped(parameters, max_norm, norm_type=2.0, **kwargs):
        if not _LOCAL_SHARDS:
            return original(parameters, max_norm, norm_type, **kwargs)

        if not _SYNCED["flag"]:
            _sync_grads()

        local = [lp for _, lp in _LOCAL_SHARDS.values()]

        grads = [p.grad for p in local if p.grad is not None]
        total_norm = _global_norm(grads, float(norm_type))
        if total_norm is None:
            return original(parameters, max_norm, norm_type, **kwargs)

        # 与 torch.clip_grad_norm_ 一致：clamp(max=1.0) 而不是 if < 1.0，
        # 这样 NaN 范数会像官方一样把梯度乘成 NaN（更早暴露），而不是静默跳过裁剪。
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        torch._foreach_mul_(grads, clip_coef_clamped)
        return total_norm

    return wrapped
