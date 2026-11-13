# Copyright 2026 Robbyant Team and/or its affiliates
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

"""Muon 的编排层：只覆盖官方 DistributedMuon 的 step 路径。

这里只替换三类优化的落地方式：
elementwise操作进行融合（muon_fused_kernels）
小矩阵打包进行NS（muon_batched_ns）
免 cat gather（muon_gather）

"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor

from lingbotvla.optim.muon import (
    DEFAULT_NS_COEFFICIENTS,
    DEFAULT_NS_STEPS,
    EPS,
    _KIND_FSDP_GATHER_2D,
    _KIND_LOCAL,
    _KIND_MOE_GATHER_3D,
    _KIND_MOE_LOCAL_3D,
    _adjust_lr,
    _classify_param,
    _full_grad,
    _get_dtensor_shard_info,
)

from .muon_batched_ns import (
    _iterate_newton_schulz,
    _megabatch_chunk_size,
    _prepare_newton_schulz,
    batched_newton_schulz,
)
from .muon_fused_kernels import (
    _FUSE_MIN_ELEMS,
    _HAS_TRITON,
    _can_fuse_apply,
    can_fuse,
    can_fuse_multi,
    momentum_nesterov_,
    muon_apply_,
    muon_apply_multi_,
)
from .muon_gather import gather_updates


_LOCAL_KINDS = (_KIND_LOCAL, _KIND_MOE_LOCAL_3D)
_NS_BATCH_MAX_BYTES = 1 << 30
_NS_BATCH_MAX_PARAM_BYTES = 64 << 20


class MuonStep:
    """
    覆盖官方 DistributedMuon 的 step。
    """

    # ------------------------------------------------------------------ 主入口

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            config = {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "momentum": float(group["momentum"]),
                "nesterov": bool(group["nesterov"]),
                "ns_coefficients": tuple(group["ns_coefficients"]),
                "ns_steps": int(group["ns_steps"]),
                "eps": float(group["eps"]),
                "adjust_lr_fn": group["adjust_lr_fn"],
            }
            fsdp_2d, others = self._partition(group["params"])

            for key in sorted(fsdp_2d):
                self._step_megabatch(fsdp_2d[key], config)

            local, singles = self._bucket(others)
            for key in sorted(local, key=lambda k: (k[0], k[1], str(k[2]))):
                self._step_local_batched(local[key], config)

            for p, kind in singles:
                self._step_single(p, kind, config)
        return loss

    # ------------------------------------------------------------ 参数分组

    @staticmethod
    def _partition(params):
        """
        按 kind 分类；FSDP 2D 按 (global shape, dtype) 分组，所有 rank 必须发同样的 collective。
        """
        fsdp_2d: Dict[tuple, List[Tensor]] = defaultdict(list)
        others: List[Tuple[Tensor, str]] = []
        for p in params:
            kind = _classify_param(p)
            if kind == _KIND_FSDP_GATHER_2D:
                fsdp_2d[(tuple(p.shape), str(p.dtype))].append(p)
                continue
            if p.grad is None:
                continue
            if torch.is_complex(p):
                raise RuntimeError("DistributedMuon does not support complex parameters")
            if p.grad.is_sparse:
                raise RuntimeError("DistributedMuon does not support sparse gradients")
            others.append((p, kind))
        return fsdp_2d, others

    @staticmethod
    def _bucket(others):
        """同 shape 的 shard-local 参数合批，其余逐参数处理。"""
        local: Dict[tuple, List[Tensor]] = defaultdict(list)
        singles: List[Tuple[Tensor, str]] = []
        for p, kind in others:
            small = p.numel() * p.element_size() <= _NS_BATCH_MAX_PARAM_BYTES
            if kind in _LOCAL_KINDS and small:
                local[(kind, tuple(p.shape), p.dtype)].append(p)
            else:
                singles.append((p, kind))
        return local, singles

    @torch.no_grad()
    def _step_megabatch(self, params: List[Tensor], config: dict) -> None:
        """同 shape 的 FSDP_GATHER_2D 参数：一次 gather + 一次 NS。"""
        if not params:
            return
        info = _get_dtensor_shard_info(params[0])
        chunk = _megabatch_chunk_size(*params[0].shape[-2:])
        for start in range(0, len(params), chunk):
            self._step_megabatch_chunk(params[start:start + chunk], info, config)

    @torch.no_grad()
    def _step_megabatch_chunk(self, params: List[Tensor], info: tuple, config: dict) -> None:
        """一个 chunk 的 megabatch 主流程：momentum → gather → NS → apply。"""
        momentum, nesterov = config["momentum"], config["nesterov"]
        pg, world_size, rank, shard_dim = info
        gather_dim = shard_dim + 1
        p0_local = self._local(params[0])
        ref_grad = params[0].grad
        update_dtype = ref_grad.dtype if ref_grad is not None else p0_local.dtype
        stacked_local = torch.empty(
            (len(params),) + tuple(p0_local.shape), dtype=update_dtype, device=p0_local.device
        )
        has_grad = [p.grad is not None for p in params]
        for idx, p in enumerate(params):
            if not has_grad[idx]:
                stacked_local[idx].zero_()
                continue
            buf_local = self._buffer(p)
            grad_local = self._local(p.grad)
            if grad_local.numel() >= _FUSE_MIN_ELEMS and can_fuse(
                grad_local, buf_local, stacked_local[idx]
            ):
                momentum_nesterov_(
                    grad_local, buf_local, stacked_local[idx], momentum, nesterov
                )
                continue
            buf_local.lerp_(grad_local, 1 - momentum)
            stacked_local[idx].copy_(
                torch.lerp(grad_local, buf_local, momentum) if nesterov else buf_local
            )
        global_size = params[0].shape[shard_dim]
        local_size = stacked_local.size(gather_dim)
        stacked_full = gather_updates(
            stacked_local, global_size=global_size, shard_dim=shard_dim,
            world_size=world_size, rank=rank, group=pg,
        )
        del stacked_local
        owner_enabled = (
            world_size > 1
            and len(params) >= world_size
            and min(params[0].shape[-2:]) >= world_size
            and global_size % world_size == 0
            and params[0].numel() >= self._owner_ns_min_matrix_elements
        )
        if owner_enabled:
            from .muon_owner_ns import partition_range, exchange_result

            owner_start, owner_count = partition_range(len(params), world_size, rank)
            owner_input = stacked_full.narrow(0, owner_start, owner_count).contiguous()
            prepared = self._prepare_ns(owner_input, config)
            del owner_input, stacked_full
            owned = _iterate_newton_schulz(
                prepared, config["ns_coefficients"], config["ns_steps"], config["eps"]
            )
            orthos_full = exchange_result(
                owned, matrix_count=len(params), world_size=world_size,
                rank=rank, shard_dim=shard_dim, group=pg,
            )
            del owned
        else:
            chunk_floor, remainder = divmod(global_size, world_size)
            shard_start = rank * chunk_floor + min(rank, remainder)
            prepared = self._prepare_ns(stacked_full, config)
            del stacked_full
            orthos_full = _iterate_newton_schulz(
                prepared, config["ns_coefficients"], config["ns_steps"], config["eps"],
                local_slice=(gather_dim, shard_start, local_size),
            )
        active = [i for i, ok in enumerate(has_grad) if ok]
        self._apply_batch([params[i] for i in active], [orthos_full[i] for i in active], config)

    @torch.no_grad()
    def _step_local_batched(self, params: List[Tensor], config: dict) -> None:
        """同 shape 本地参数：按字节切块，每块一次 batched NS + 一次 apply。"""
        if not params:
            return
        nbytes = params[0].numel() * params[0].element_size()
        per_chunk = max(1, int(_NS_BATCH_MAX_BYTES // max(1, nbytes)))
        for start in range(0, len(params), per_chunk):
            chunk = params[start:start + per_chunk]
            updates = [self._momentum_update(p, config) for p in chunk]
            if len(updates) == 1:
                orthos = [self._ns(updates[0], config)]
            else:
                stacked = self._ns(torch.stack(updates, dim=0), config)
                del updates
                orthos = list(stacked.unbind(0))
            self._apply_batch(chunk, orthos, config)


    @torch.no_grad()
    def _step_single(self, p: Tensor, kind: str, config: dict) -> None:
        """单参数路径：需要完整矩阵的 3D MoE 分片走这里。"""
        update = self._momentum_update(p, config)
        if kind == _KIND_MOE_GATHER_3D:
            assert isinstance(p, DTensor)
            full = _full_grad(DTensor.from_local(
                update, device_mesh=p.device_mesh,
                placements=p.placements, run_check=False,
            ))
            full_ortho = self._ns(full, config)
            _, world_size, rank, shard_dim = _get_dtensor_shard_info(p)
            chunk_floor, remainder = divmod(p.shape[shard_dim], world_size)
            start = rank * chunk_floor + min(rank, remainder)
            ortho = full_ortho.narrow(
                shard_dim, start, update.shape[shard_dim]
            ).contiguous()
        else:
            ortho = self._ns(update, config)
        self._apply_batch([p], [ortho], config)

    # ------------------------------------------------------------ 公共底层

    @torch.no_grad()
    def _apply_batch(self, params: List[Tensor], orthos: List[Tensor], config: dict) -> None:
        """整组一次 multi-tensor apply。"""
        if not params:
            return
        lr, weight_decay = config["lr"], config["weight_decay"]
        decay = 1 - lr * weight_decay
        adjusted_lr = _adjust_lr(lr, config["adjust_lr_fn"], params[0].shape[-2:])

        if len(params) == 1 and _can_fuse_apply(self._local(params[0]), orthos[0]):
            muon_apply_(self._local(params[0]), orthos[0], decay, adjusted_lr)
            return

        fused_params: List[Tensor] = []
        fused_orthos: List[Tensor] = []
        for p, ortho in zip(params, orthos):
            p_local = self._local(p)
            if not _can_fuse_apply(p_local, ortho):
                break
            fused_params.append(p_local)
            fused_orthos.append(ortho)
        if (
            fused_params
            and len(fused_params) == len(params)
            and can_fuse_multi(fused_params, torch.float32)
            and can_fuse_multi(fused_orthos, fused_orthos[0].dtype)
        ):
            muon_apply_multi_(fused_params, fused_orthos, decay, adjusted_lr)
            return

        for p, ortho in zip(params, orthos):
            p_local = self._local(p)
            if weight_decay != 0.0:
                p_local.mul_(decay)
            p_local.add_(ortho.to(dtype=p_local.dtype), alpha=-adjusted_lr)

    @staticmethod
    def _local(t: Tensor) -> Tensor:
        return t.to_local() if isinstance(t, DTensor) else t

    def _buffer(self, p: Tensor) -> Tensor:
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                p, memory_format=torch.preserve_format
            )
        return self._local(state["momentum_buffer"])

    def _momentum_update(self, p: Tensor, config: dict) -> Tensor:
        """更新 momentum buffer，返回本次的 update。"""
        buf_local = self._buffer(p)
        grad_local = self._local(p.grad)
        buf_local.lerp_(grad_local, 1 - config["momentum"])
        if config["nesterov"]:
            return grad_local.lerp(buf_local, config["momentum"])
        return buf_local.clone()

    def _prepare_ns(self, grad: Tensor, config: dict):
        return _prepare_newton_schulz(
            grad, config["ns_coefficients"], config["ns_steps"],
            torch.bfloat16, layout_pack=True,
        )

    def _ns(self, grad: Tensor, config: dict, local_slice=None) -> Tensor:
        return batched_newton_schulz(
            grad, config["ns_coefficients"], config["ns_steps"], config["eps"],
            layout_pack=True, local_slice=local_slice,
        )


def configure_optimizer(original, _options=None):
    if not _HAS_TRITON:
        raise RuntimeError("lingbot-vla-v2.muon-fused requires triton")
    cls = type("DistributedMuon", (MuonStep, original), {
        "_owner_ns_min_matrix_elements": 1 << 20,
    })
    cls.__module__ = __name__
    cls.__qualname__ = "DistributedMuon"
    return cls


__all__ = [
    "DEFAULT_NS_COEFFICIENTS",
    "DEFAULT_NS_STEPS",
    "batched_newton_schulz",
    "configure_optimizer",
]
