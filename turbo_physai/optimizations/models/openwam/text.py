# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main 0c9ed80d80af1d0c5f4148afe65cc7f462ec3995.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for text encoding and collating."""

import logging
import torch

logger = logging.getLogger(__name__)


def collate_cpu_padding(samples):
    """Keep tiny proprio masks as NumPy metadata through Accelerate's transfer.

    _collect_inputs converts these back to CPU bool tensors. All other sample
    fields retain the original list collator's behavior, including pinning and
    nonblocking transfer of large tensors. This top-level collator is picklable.
    """
    result = []
    for sample in samples:
        mask = sample.get("proprio_mask")
        if isinstance(mask, torch.Tensor) and mask.device.type == "cpu":
            sample = dict(sample)
            sample["proprio_mask"] = mask.detach().to(dtype=torch.bool).numpy().copy()
        result.append(sample)
    return result


def collate_cpu_padding_uint8(samples):
    """Chain collate_cpu_padding with openwam_collate for multi-worker picklability."""
    from openwam.dataloader.collate import openwam_collate

    return openwam_collate(collate_cpu_padding(samples))


def frozen_eval(encoder):
    return not any(m.training for m in encoder.modules()) and not any(
        p.requires_grad for p in encoder.parameters()
    )


def _native_t5_forward(module):
    forward = module.forward
    if getattr(module, "_openwam_no_grad_wrapped", False):
        from openwam.model.architectures.base import _wrap_single_forward

        code = getattr(forward, "__code__", None)
        if (
            not any(
                code is value
                for value in _wrap_single_forward.__code__.co_consts
                if getattr(value, "co_name", None) == "wrapped"
            )
            or getattr(forward, "__globals__", None)
            is not _wrap_single_forward.__globals__
            or code.co_freevars != ("original_forward",)
        ):
            return False
        closure = getattr(forward, "__closure__", None)
        if not closure or len(closure) != 1:
            return False
        try:
            original = closure[0].cell_contents
        except ValueError:
            return False
        if getattr(forward, "__wrapped__", None) is not original:
            return False
        forward = original
    return (
        getattr(forward, "__self__", None) is module
        and getattr(forward, "__func__", None) is type(module).forward
    )


def _native_t5(encoder):
    from openwam.model.video_backbone.wan.models import text_encoder as native

    kinds = (
        native.WanTextEncoder,
        native.T5Attention,
        native.T5FeedForward,
        native.T5LayerNorm,
        native.T5RelativeEmbedding,
        native.T5SelfAttention,
        native.GELU,
        torch.nn.Embedding,
        torch.nn.Linear,
        torch.nn.Dropout,
        torch.nn.ModuleList,
        torch.nn.Sequential,
    )
    hooks = (
        "_forward_pre_hooks",
        "_forward_hooks",
        "_backward_pre_hooks",
        "_backward_hooks",
    )
    if type(encoder) is not native.WanTextEncoder:
        return False
    if any(getattr(torch.nn.modules.module, "_global" + name, None) for name in hooks):
        return False
    return all(
        type(module) in kinds
        and _native_t5_forward(module)
        and getattr(module, "_compiled_call_impl", None) is None
        and not any(getattr(module, name, None) for name in hooks)
        for module in encoder.modules()
    )
