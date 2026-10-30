# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA FSDP1 逐层 ``torch.compile``。

解决直接对整个模型使用torch.compile在 OpenVLA 上不生效的问题：
追踪 ``LlamaModel.forward`` 时会把层循环吸进同一个 frame，循环体里的 ``torch.utils.checkpoint`` HOP 无法被 Dynamo内省，
整个 frame 被跳过，LLM 静默退回 eager。

本 Group 改为逐个 FSDP 单元独立 compile：

    FSDP(layer) -> torch.compile(fsdp_layer) -> checkpoint_wrapper(compiled)
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.optimization import (
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
)


def _disabled() -> bool:
    """仓库约定的统一开关：置 1 时跳过本组件内全部 ``torch.compile``。"""

    return os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1"


MODE = "default"


def _fsdp_class() -> type:
    """调用时动态解析 FSDP1 类，让 ``openvla.fsdp.prefetch`` 的注入生效。"""

    from torch.distributed.fsdp import FullyShardedDataParallel

    return FullyShardedDataParallel


def _replace_child(module: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(module, nn.Sequential):
        module[int(name)] = child
    else:
        setattr(module, name, child)


def _per_layer_compile_and_checkpoint(module, layer_cls, mode, checkpoint) -> None:
    """对每个 FSDP 包住的 ``layer_cls`` 先 compile 再套 checkpoint。"""

    for name, child in list(module.named_children()):
        if isinstance(child, FSDP) and isinstance(child._fsdp_wrapped_module, layer_cls):
            child = torch.compile(child, mode=mode, fullgraph=False, dynamic=True)
            if checkpoint:
                child = checkpoint_wrapper(
                    child, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
            _replace_child(module, name, child)
        else:
            _per_layer_compile_and_checkpoint(child, layer_cls, mode, checkpoint)


def _compile_vision_towers(vision_backbone, mode) -> None:
    """ViT整体 compile。"""

    for name, child in list(vision_backbone.named_children()):
        if isinstance(child, FSDP) and isinstance(
            child._fsdp_wrapped_module, VisionTransformer
        ):
            _replace_child(
                vision_backbone,
                name,
                torch.compile(child, mode=mode, fullgraph=False, dynamic=True),
            )


def _param_groups(vlm, weight_decay):
    """bias / norm 参数不 decay。"""

    decay, no_decay = [], []
    for name, param in vlm.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run_setup_wrapper(original: Callable, options) -> Callable:
    """wrap 工厂：复现 baseline ``run_setup``，外加逐层 compile。"""

    del options

    if _disabled():
        return original
    mode = MODE

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        policy = self.vlm.get_fsdp_wrapping_policy()

        if (
            self.enable_mixed_precision_training
            and self.mixed_precision_dtype == torch.bfloat16
        ):
            reduce_dtype = (
                torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            )
            # compile on -> buffer_dtype=None，避免 Dynamo fake-tensor set_ 报错。
            precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_dtype,
                buffer_dtype=None,
            )
            if self.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
                self.vlm.vision_backbone.to(
                    dtype=self.vlm.vision_backbone.half_precision_dtype
                )
        else:
            precision = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )

        # 不传 limit_all_gathers：默认 True 保持 baseline 语义，同时让
        # openvla.fsdp.prefetch 的 kwargs.setdefault 能覆盖它。
        self.vlm = _fsdp_class()(
            self.vlm,
            auto_wrap_policy=policy,
            mixed_precision=precision,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

        # FSDP(layer) -> torch.compile(fsdp_layer) -> checkpoint_wrapper(compiled)
        _per_layer_compile_and_checkpoint(
            self.vlm,
            self.llm_transformer_layer_cls,
            mode,
            self.enable_gradient_checkpointing,
        )
        _compile_vision_towers(self.vlm.vision_backbone, mode)
        dist.barrier()

        n_train_examples = (
            math.ceil(n_train_examples / self.global_batch_size)
            * self.global_batch_size
        )
        steps = (
            self.max_steps
            if self.max_steps is not None
            else (n_train_examples * self.epochs) // self.global_batch_size
        )

        # 动态解析 torch.optim.AdamW，让 openvla.adamw.fused 的 fused 默认值生效。
        groups = _param_groups(self.vlm, self.weight_decay)
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            self.optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, int(steps * self.warmup_ratio), steps
            )
            for group in self.optimizer.param_groups:
                group["lr"] = 0.0
        elif self.lr_scheduler_type == "constant":
            self.optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(
                f"unsupported lr_scheduler_type: {self.lr_scheduler_type!r}"
            )

    return run_setup


def _partition_state_dicts(state_dict, module_keys):
    """按顶层模块前缀切分 full state dict，并剥掉 ``._orig_mod.``。"""

    result = {key: OrderedDict() for key in module_keys}
    for name, param in state_dict.items():
        name = name.replace("._orig_mod.", ".")
        for prefix in module_keys:
            if name.startswith(prefix + "."):
                result[prefix][name[len(prefix) + 1 :]] = param
                break
    return result


def save_checkpoint_wrapper(original: Callable, options) -> Callable:
    """``wrap`` 工厂：复现 baseline ``save_checkpoint``，外加剥离 ``._orig_mod.``。"""

    del options

    if _disabled():
        return original

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss=None,
        only_trainable: bool = True,
    ) -> None:
        vlm = getattr(self.vlm, "_orig_mod", self.vlm)
        assert isinstance(vlm, FSDP), "VLM must be FSDP-wrapped before saving"

        rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
        with FSDP.state_dict_type(vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            state = _partition_state_dicts(
                vlm.state_dict(),
                self.trainable_module_keys if only_trainable else self.all_module_keys,
            )
            if rank == 0:
                loss = "inf" if train_loss is None else f"{train_loss:.4f}"
                torch.save(
                    {"model": state},
                    run_dir
                    / "checkpoints"
                    / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={loss}.pt",
                )

    return save_checkpoint
