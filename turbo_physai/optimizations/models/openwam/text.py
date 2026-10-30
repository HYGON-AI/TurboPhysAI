# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Bounded prompt caching restricted to frozen, deterministic text encoders."""

from turbo_physai.optimizations.models.openwam.settings import value

import logging
from collections import OrderedDict

import torch

from turbo_physai.optimizations.models.openwam import enabled

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


def t5_prefix_plan(ids, mask, text_encoder):
    """Return a CPU right-padding plan for the native frozen T5, or None."""
    if not enabled("t5_prefix") or not frozen_eval(text_encoder):
        return None
    if (
        not _native_t5(text_encoder)
        or ids.device.type != "cpu"
        or mask.device.type != "cpu"
        or ids.ndim != 2
        or mask.shape != ids.shape
        or not ids.shape[0]
        or not ids.shape[1]
    ):
        return None
    if not bool(((mask == 0) | (mask == 1)).all()):
        return None
    keep = mask > 0
    lengths = keep.sum(dim=1, dtype=torch.long)
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    if not torch.equal(keep, positions < lengths.unsqueeze(1)):
        return None
    maximum = int(lengths.max())
    # Keep the original all-empty route: T5 implementations need a nonempty axis.
    if maximum == 0 or maximum == ids.shape[1]:
        return None
    return maximum, lengths.tolist()


def trim_text_context(inputs):
    """Crop only the batch's common masked tail, before any proprio append."""
    if not enabled("text_trim"):
        return inputs
    context = inputs.get("context")
    if (
        not isinstance(context, torch.Tensor)
        or context.ndim != 3
        or context.shape[1] <= 1
    ):
        return inputs
    mask = inputs.get("context_mask")
    lengths = inputs.get("seq_lens")
    if mask is not None:
        # A custom/additive or query-dependent mask is not a pure padding mask.
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != context.shape[:2]
        ):
            return inputs
        keep_cpu = mask.detach().to(device="cpu")
        columns = keep_cpu.any(dim=0).nonzero(as_tuple=False).flatten()
        stop = int(columns[-1]) + 1 if columns.numel() else 1
    elif isinstance(lengths, torch.Tensor) and lengths.shape == (context.shape[0],):
        if lengths.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return inputs
        lengths_cpu = lengths.detach().to(device="cpu")
        if lengths_cpu.numel() == 0 or bool(
            ((lengths_cpu < 0) | (lengths_cpu > context.shape[1])).any()
        ):
            return inputs
        stop = max(1, int(lengths_cpu.max()))
    else:
        return inputs
    if stop >= context.shape[1]:
        return inputs
    # If both representations exist, retain their range invariant. seq_lens
    # stays unchanged, and an explicit mask remains the source of visibility.
    if isinstance(lengths, torch.Tensor):
        if lengths.shape != (context.shape[0],):
            return inputs
        lengths_cpu = lengths.detach().to(device="cpu")
        if bool(((lengths_cpu < 0) | (lengths_cpu > stop)).any()):
            return inputs
    result = dict(inputs)
    result["context"] = context[:, :stop].contiguous()
    if mask is not None:
        result["context_mask"] = mask[:, :stop].contiguous()
    # Keep one masked slot for all-empty batches, so native attention never
    # receives a zero-length KV axis. No valid token or interior hole is removed.
    return result


def cached_text(prompts, *, tokenizer, text_encoder, device, encode):
    if not enabled("text_cache"):
        return encode(prompts)
    store = getattr(text_encoder, "_openwam_text_store", None)
    if store is not None:
        return store(prompts, device)
    from turbo_physai.optimizations.models.openwam.text_store import requested

    if requested():
        raise RuntimeError(
            "Offline T5 cache requested but not attached; initialize it before encoding training text"
        )
    params = tuple(text_encoder.parameters())
    if not frozen_eval(text_encoder):
        if not getattr(text_encoder, "_openwam_cache_skip_warned", False):
            logger.warning(
                "[optimizations] text cache skipped: encoder must be frozen and in eval mode"
            )
            text_encoder._openwam_cache_skip_warned = True
        return encode(prompts)
    capacity = int(value("text_cache_size", "32"))
    if capacity <= 0:
        raise ValueError("text_cache_size must be positive")
    signature = (
        id(tokenizer),
        str(device),
        enabled("t5_prefix"),
        enabled("padding_plan_cpu"),
        tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in params),
    )
    if signature != getattr(text_encoder, "_openwam_text_cache_signature", None):
        text_encoder._openwam_text_cache_signature = signature
        text_encoder._openwam_text_cache = OrderedDict()
    cache = text_encoder._openwam_text_cache
    unique = list(dict.fromkeys(prompts))
    # Keep this batch's references separately so eviction cannot invalidate it.
    values = {prompt: cache[prompt] for prompt in unique if prompt in cache}
    missing = [prompt for prompt in unique if prompt not in values]
    if missing:
        with torch.no_grad():
            context, lengths = encode(missing)
        for index, prompt in enumerate(missing):
            values[prompt] = (
                context[index].detach().clone(),
                lengths[index].detach().clone(),
            )
    for prompt in unique:
        cache[prompt] = values[prompt]
        cache.move_to_end(prompt)
    while len(cache) > capacity:
        cache.popitem(last=False)
    return torch.stack([values[prompt][0] for prompt in prompts]), torch.stack(
        [values[prompt][1] for prompt in prompts]
    )
