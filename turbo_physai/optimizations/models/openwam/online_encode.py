# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main ac8aab4f6ba16b1739c94df823cc444478e9dab0.
# SPDX-License-Identifier: Apache-2.0

"""One real batch of lookahead encoding during the ZeRO-2 parameter AllGather."""

import functools
import inspect
import logging
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist

from turbo_physai.optimizations.models.openwam.zero import validate_zero

logger = logging.getLogger(__name__)


@dataclass
class PreparedInputs:
    values: dict


class OnlineEncoder:
    """Instance-local gather replacement; no threads, side streams, replay, or disk cache.

    The AllGather is launched asynchronously, the next batch is encoded on the
    training stream, and only then does the training stream wait for the
    collective. Encoding runs on the compute units while RCCL gathers the
    updated parameters. Frozen encoders are not in the gathered groups, so the
    encode never reads a buffer the collective writes. A single stream keeps
    one allocator pool and needs no events or record_stream.
    """

    def __init__(
        self,
        architecture,
        optimizer,
        *,
        num_workers,
        gradient_accumulation_steps,
        deferred=None,
    ):
        from turbo_physai.optimizations.models.openwam.text import (
            _native_t5,
            frozen_eval,
        )

        validate_zero(optimizer)
        bb = architecture.video_backbone
        if (
            type(bb).__name__ != "Wan22Ti2v"
            or bb.video_encoder is not None
            or type(bb.vae).__name__ != "WanVideoVAE38"
            or not _native_t5(bb.text_encoder)
        ):
            raise ValueError(
                "Online encoding currently requires native Wan22Ti2v VAE38 and T5"
            )
        if not all(frozen_eval(module) for module in (bb.vae, bb.text_encoder)):
            raise ValueError("Online encoding requires frozen VAE/T5 in eval mode")
        frozen_ids = {
            id(p) for module in (bb.vae, bb.text_encoder) for p in module.parameters()
        }
        if any(id(p) in frozen_ids for group in optimizer.bit16_groups for p in group):
            raise ValueError(
                "Frozen encoders must not belong to the ZeRO optimizer/AllGather groups"
            )
        if gradient_accumulation_steps != 1 or num_workers < 1:
            raise ValueError(
                "Online encoding requires gradient accumulation=1 and DataLoader workers>0"
            )
        # Worker-side random transforms keep their own RNG/order. Native TI2V
        # preprocessing uses the VAE mean and no random draws; training noise
        # and timesteps remain in compute_loss after the per-step seed.
        self.architecture = architecture
        self.optimizer = optimizer
        self.deferred = deferred
        self.pending = None
        self.next_batch = None
        self.original_step = optimizer.step
        function = self.original_step.__func__
        if "all_gather_dp_groups" not in function.__code__.co_names:
            raise RuntimeError(
                "Unsupported DeepSpeed step: no direct all_gather_dp_groups call"
            )
        self.original_gather = function.__globals__["all_gather_dp_groups"]
        expected = (
            "groups_flat",
            "partitioned_param_groups",
            "dp_process_group",
            "start_alignment_factor",
            "allgather_bucket_size",
        )
        if tuple(inspect.signature(self.original_gather).parameters) != expected:
            raise RuntimeError("Unsupported DeepSpeed AllGather interface")
        namespace = dict(function.__globals__)
        namespace["all_gather_dp_groups"] = self._gather
        replacement = types.FunctionType(
            function.__code__,
            namespace,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        replacement.__kwdefaults__ = function.__kwdefaults__
        functools.update_wrapper(replacement, function)
        self.step = types.MethodType(replacement, optimizer)

    def __enter__(self):
        self.optimizer.step = self.step
        return self

    def __exit__(self, *exc):
        self.optimizer.step = self.original_step
        self.next_batch = None
        self.pending = None

    def _prepare_lookahead(self, batch):
        # Reorder only after gather launches. Initial/fallback batches retain
        # the native T5-first allocation and compilation order.
        backbone = self.architecture.video_backbone
        missing = object()
        previous = backbone.__dict__.get("_openwam_video_first", missing)
        backbone._openwam_video_first = True
        try:
            with torch.no_grad():
                values = self.architecture.prepare_inputs(
                    batch if isinstance(batch, list) else [batch]
                )
            return PreparedInputs(values)
        finally:
            if previous is missing:
                del backbone._openwam_video_first
            else:
                backbone._openwam_video_first = previous

    def _gather(
        self,
        groups_flat,
        partitioned_param_groups,
        dp_process_group,
        start_alignment_factor,
        allgather_bucket_size,
    ):
        if self.deferred is not None:
            # deferred_gather: launch the owner broadcasts and encode
            # behind them; the next forward waits per layer, not this step.
            self.deferred.gather(
                groups_flat, partitioned_param_groups, dp_process_group
            )
            if self.next_batch is not None:
                take, self.next_batch = self.next_batch, None
                batch = take()
                self.pending = self._prepare_lookahead(batch)
            return
        if self.next_batch is None:
            return self.original_gather(
                groups_flat,
                partitioned_param_groups,
                dp_process_group,
                start_alignment_factor,
                allgather_bucket_size,
            )
        take, self.next_batch = self.next_batch, None
        works = []
        try:
            for flat, parts, group in zip(
                groups_flat, partitioned_param_groups, dp_process_group
            ):
                if dist.get_world_size(group) > 1:
                    works.append(
                        dist.all_gather_into_tensor(
                            flat,
                            parts[dist.get_rank(group)],
                            group=group,
                            async_op=True,
                        )
                    )
            # The collective has captured its dependency on the optimizer update;
            # everything enqueued below runs on the compute stream alongside it.
            batch = take()
            self.pending = self._prepare_lookahead(batch)
        finally:
            # Work.wait makes the training stream wait for the gathered
            # parameters; it must follow the encode so the two overlap. It also
            # runs if fetching fails, so no collective is left outstanding.
            for work in works:
                work.wait()

    def batches(self, batches, *, global_step, max_steps, save_steps):
        # Pinned batches are copied host-to-device asynchronously on the training
        # stream; the encode that consumes them is ordered after the copy.
        if getattr(batches, "_non_blocking", None) is False:
            batches._non_blocking = True
        iterator = iter(batches)
        try:
            for index in range(len(batches)):
                if self.pending is None:
                    batch = next(iterator)
                else:
                    batch, self.pending = self.pending, None
                step = global_step + index + 1
                eligible = (
                    index + 1 < len(batches)
                    and (not max_steps or step < max_steps)
                    and (not save_steps or step % save_steps != 0)
                )
                self.next_batch = (lambda: next(iterator)) if eligible else None
                yield batch
                # Overflow can skip optimizer.step/AllGather. In that case no
                # next batch was consumed, so the next iteration reads it here.
                self.next_batch = None
        finally:
            self.next_batch = None
