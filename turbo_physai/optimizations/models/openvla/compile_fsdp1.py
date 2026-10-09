# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenVLA FSDP1 per-layer ``torch.compile`` wrappers for TurboPhysAI.

What this does
--------------
Baseline ``prismatic.training.strategies.fsdp.FSDPStrategy`` has *no*
``torch.compile`` support at all.  Whole-model ``torch.compile`` would trace
``LlamaModel.forward`` whose layer loop calls into ``torch.utils.checkpoint``
(the checkpoint HOP); Dynamo cannot introspect that HOP inside a traced loop and
skips the whole frame (``convert_frame.py:854``), so the LLM silently falls back
to eager.

This module delivers the FSDP1 per-layer compile flavour (the same idea as
``compileable_fsdp.py`` in the working tree) by **wrapping two methods that DO
exist on the baseline FSDPStrategy**:

  * ``FSDPStrategy.run_setup``      -> ``run_setup_wrapper`` (compile branch)
  * ``FSDPStrategy.save_checkpoint``-> ``save_checkpoint_wrapper``
                                    (strips ``_orig_mod``)

Each wrapper reads the ``openvla.compile.fsdp1`` Group ``options``.  The
on branch (when ``options.compile`` is true) reproduces the baseline
``run_setup`` body but:

  1. keeps ``buffer_dtype=None`` (avoids the Dynamo fake-tensor ``set_`` dtype
     error under bf16, PyTorch #152162/#161153);
  2. instead of ``apply_activation_checkpointing`` (which puts the checkpoint HOP
     *inside* the compiled unit), compiles each FSDP1-wrapped LLM transformer
     layer individually and puts ``checkpoint_wrapper`` OUTSIDE the compiled unit;
  3. whole-model-compiles each FSDP1-wrapped vision tower (no checkpoint HOP
     lives inside a tower, so a whole-model compile is safe there).

Switch
------
Whether the compile branch is used is decided by the ``openvla.compile.fsdp1``
Group ``options`` in the OptimizationConfig / recipe -- **not** by any
environment variable:

* ``options.compile`` (bool, default ``False``): enable per-layer
  ``torch.compile``.
* ``options.mode`` (str, default ``"default"``): the ``torch.compile`` mode
  (``default`` / ``reduce-overhead`` / ``max-autotune``).

Each replacement is declared as a ``wrap`` whose factory reads those options.
When ``options.compile`` is ``False`` the factory returns the *original*
baseline method untouched, so the run is exactly the un-optimized baseline
(nothing is compiled and nothing is patched).  Compile is therefore an explicit
opt-in via config rather than a silent environment default.

Note on imports
---------------
This module is only imported when TurboPhysAI applies the Group, i.e. inside the
real training subprocess launched by ``turbo-physai run`` (baseline on
``sys.path``), so top-level ``import prismatic`` would be safe here.  In practice
the branch bodies only ever act on an existing ``FSDPStrategy`` instance via
``self`` + torch/fsdp/timm, so they do not need any ``prismatic`` symbol; we
therefore keep the module importable anywhere (no prismatic dependency at all),
which sidesteps interpreter-startup ordering entirely.
"""

from __future__ import annotations

import logging
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

# A plain logger (rank-agnostic; informational only).  The baseline overwatch
# logger is not needed to reproduce the compile behaviour.
logger = logging.getLogger("openvla.compile.fsdp1")


# --- runtime switch (config-driven) ---------------------------------------
# Compile is OFF unless the Group explicitly opts in via its ``options``.
# See the module docstring for ``compile`` / ``mode`` semantics.


def _compile_options(options) -> tuple[bool, str]:
    """Return ``(enabled, mode)`` from the Group ``options`` mapping."""
    options = dict(options or {})
    enabled = bool(options.get("compile", False))
    mode = str(options.get("mode", "default") or "default")
    return enabled, mode


def _fsdp_class() -> type:
    """Resolve the FSDP1 class to *construct* with, at call time.

    The module-level ``FSDP`` name above is bound when this module is imported,
    and TurboPhysAI imports every replacement module during its preparation
    phase -- i.e. **before** any Group is installed.  So that binding is always
    the *unpatched* class and must not be used to construct: the
    ``openvla.fsdp.prefetch`` Group replaces
    ``torch.distributed.fsdp.FullyShardedDataParallel`` with a subclass that
    injects ``limit_all_gathers`` / ``forward_prefetch`` / ``backward_prefetch``,
    and only a dynamic lookup sees it.  Without this, enabling both Groups would
    silently drop the prefetch optimization whenever compile is on.

    The module-level ``FSDP`` binding is still the right object for the
    ``isinstance`` / ``assert isinstance`` checks below: an instance built from
    the subclass satisfies them too, whether or not the patch is installed.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel

    return FullyShardedDataParallel


# --- per-layer compile helpers ---------------------------------------------
def _replace_child(module: nn.Module, name: str, child: nn.Module) -> None:
    """Replace a child submodule (mirrors ``apply_activation_checkpointing``)."""
    if isinstance(module, nn.Sequential):
        module[int(name)] = child
    else:
        setattr(module, name, child)


def _per_layer_compile_and_checkpoint(
    module: nn.Module,
    layer_cls: type,
    compile_mode: Optional[str],
    checkpoint_module: bool,
) -> None:
    """DFS: for every FSDP1-wrapped submodule of ``layer_cls`` compile + checkpoint."""
    for name, child in list(module.named_children()):
        if isinstance(child, FSDP) and isinstance(child._fsdp_wrapped_module, layer_cls):
            if compile_mode is not None:
                child = torch.compile(child, mode=compile_mode, fullgraph=False, dynamic=True)
            if checkpoint_module:
                child = checkpoint_wrapper(child, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            _replace_child(module, name, child)
        else:
            _per_layer_compile_and_checkpoint(child, layer_cls, compile_mode, checkpoint_module)


def _compile_whole_vision_towers(vision_backbone: nn.Module, compile_mode: Optional[str]) -> None:
    """Whole-model ``torch.compile`` each FSDP-wrapped ViT tower."""
    if compile_mode is None:
        return
    for name, child in list(vision_backbone.named_children()):
        if isinstance(child, FSDP) and isinstance(child._fsdp_wrapped_module, VisionTransformer):
            _replace_child(
                vision_backbone,
                name,
                torch.compile(child, mode=compile_mode, fullgraph=False, dynamic=True),
            )


# --- FSDPStrategy.run_setup wrapper -----------------------------------------
def run_setup_wrapper(original: Callable[..., Any], options) -> Callable[..., Any]:
    """wrap factory for ``FSDPStrategy.run_setup``.

    ``options.compile`` False -> return the baseline method untouched (clean
    baseline, nothing compiled).  True -> return a ``run_setup`` that reproduces
    the baseline body and additionally applies per-layer ``torch.compile`` using
    ``options.mode``.
    """
    enabled, compile_mode = _compile_options(options)
    if not enabled:
        return original

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        """FSDP run_setup with per-layer torch.compile (the on branch).

        Reproduces baseline ``FSDPStrategy.run_setup`` but sets
        ``buffer_dtype=None`` and wraps each FSDP1 LLM layer with
        ``torch.compile`` then ``checkpoint_wrapper`` OUTSIDE the compiled unit,
        and whole-model-compiles the vision towers.
        """
        vlm_fsdp_wrapping_policy = self.vlm.get_fsdp_wrapping_policy()

        # Mixed precision policy.
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            # compile on -> buffer_dtype=None to avoid Dynamo fake-tensor set_ error.
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
                buffer_dtype=None,
            )
            if self.stage not in {"full-finetune", "vla-full-train", "vla-sandwich-train"}:
                logger.info("Casting Vision Backbone to *Half Precision* via `.to(dtype=...)`")
                self.vlm.vision_backbone.to(dtype=self.vlm.vision_backbone.half_precision_dtype)
        else:
            fsdp_precision_policy = MixedPrecision(
                param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )

        # FSDP wrap (keeps baseline sharding semantics untouched).
        # `_fsdp_class()` -- resolved at call time so `openvla.fsdp.prefetch`
        # (installed on the same class attribute) is honoured here as well.
        #
        # NOTE: `limit_all_gathers` is deliberately NOT passed here.  Baseline
        # semantics are preserved either way because torch's default is `True`,
        # whereas passing it explicitly would make this call site the "explicit
        # caller" and win over the `openvla.fsdp.prefetch` injection (the
        # wrapper uses `kwargs.setdefault`), silently keeping the all-gather
        # rate limiter on for the compiled path.
        self.vlm = _fsdp_class()(
            self.vlm,
            auto_wrap_policy=vlm_fsdp_wrapping_policy,
            mixed_precision=fsdp_precision_policy,
            sharding_strategy=self.fsdp_sharding_strategy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )

        # Per-layer compile + gradient checkpointing OUTSIDE the compiled unit.
        # (FSDP(layer) -> torch.compile(fsdp_layer) -> checkpoint_wrapper(compiled)).
        _per_layer_compile_and_checkpoint(
            module=self.vlm,
            layer_cls=self.llm_transformer_layer_cls,
            compile_mode=compile_mode,
            checkpoint_module=self.enable_gradient_checkpointing,
        )

        # Whole-model compile of the FSDP-wrapped vision towers.
        _compile_whole_vision_towers(
            vision_backbone=self.vlm.vision_backbone,
            compile_mode=compile_mode,
        )

        dist.barrier()

        # Optimizer & LR scheduler (torch native AdamW, same as baseline FSDPStrategy).
        # Resolve `torch.optim.AdamW` dynamically at call time (not a module-level
        # `from torch.optim import AdamW` binding) so the `openvla.adamw.fused` wrap
        # factory -- installed on `torch.optim.AdamW` at startup -- is honoured here.
        n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
            self.optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0
        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
            self.optimizer = torch.optim.AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_constant_schedule(self.optimizer)
        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        logger.info(
            "FSDP1 compile run_setup done (mode=`%s`, sharding=`%s`). VLM FSDP = %s",
            compile_mode,
            self.fsdp_sharding_strategy,
            type(self.vlm).__name__,
        )

    return run_setup


# --- FSDPStrategy.save_checkpoint wrapper -----------------------------------
def _partition_state_dicts(
    full_state_dict,
    module_keys,
    *,
    strip_orig_mod: bool = True,
):
    """Split a flattened (FSDP full) state dict into per-module-key OrderedDicts.

    ``module_keys`` are the top-level name prefixes of the modules to keep
    (``trainable_module_keys`` or ``all_module_keys``).  When ``strip_orig_mod``
    is true, ``._orig_mod.`` segments introduced by per-layer ``torch.compile``
    are removed so the produced checkpoint keys match the plain FSDP format.

    This is exactly the mapping used by ``save_checkpoint``; exposing it makes
    the compiled-vs-plain weight consistency directly testable without FSDP/CUDA.
    """
    model_state_dicts = {mkey: OrderedDict() for mkey in module_keys}
    for key, param in full_state_dict.items():
        if strip_orig_mod:
            key = key.replace("._orig_mod.", ".")
        for mkey in module_keys:
            if key.startswith(prefix := f"{mkey}."):
                model_state_dicts[mkey][key.removeprefix(prefix)] = param
                break
    return model_state_dicts


def save_checkpoint_wrapper(original: Callable[..., Any], options) -> Callable[..., Any]:
    """wrap factory for ``FSDPStrategy.save_checkpoint``.

    ``options.compile`` False -> return the baseline method untouched (clean
    baseline).  True -> return a ``save_checkpoint`` that strips the
    per-layer-compile ``._orig_mod.`` key segments before saving.
    """
    enabled, _compile_mode = _compile_options(options)
    if not enabled:
        return original

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Baseline save_checkpoint + strip per-layer-compile ``._orig_mod.`` key segments.

        After per-layer ``torch.compile`` each compiled layer / tower is an
        ``OptimizedModule`` whose ``_orig_mod`` submodule shows up in state_dict keys
        (``...layers.0._orig_mod.self_attn...``).  We strip that segment so the saved
        checkpoint format stays byte-identical to the plain FSDP strategy.
        """
        vlm = getattr(self.vlm, "_orig_mod", self.vlm)
        assert isinstance(vlm, FSDP), "save_checkpoint assumes VLM is already wrapped in FSDP!"

        with FSDP.state_dict_type(vlm, self.fsdp_state_dict_type, self.fsdp_save_policy):
            full_vlm_state_dict = vlm.state_dict()
            model_state_dicts = _partition_state_dicts(
                full_vlm_state_dict,
                self.trainable_module_keys if only_trainable else self.all_module_keys,
                strip_orig_mod=True,
            )

            # Baseline uses a module-level overwatch logger; we reproduce rank0-only
            # saving via the distributed backend / RANK env instead.
            try:
                is_rank_zero = dist.get_rank() == 0 if dist.is_initialized() else True
            except Exception:
                is_rank_zero = int(os.environ.get("RANK", "0")) == 0

            if is_rank_zero:
                checkpoint_dir = run_dir / "checkpoints"
                if train_loss is None:
                    checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
                else:
                    checkpoint_path = (
                        checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                    )
                torch.save({"model": model_state_dicts}, checkpoint_path)

    return save_checkpoint
