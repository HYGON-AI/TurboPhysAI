# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main d1ea554ba44057b766fff94c0d999fb11a217ffb.
# SPDX-License-Identifier: Apache-2.0

"""Prepare local norms on a side stream AFTER final gradient accumulation.

No collective runs on this stream. DeepSpeed retains overflow, DP/MP norm
reduction, loss unscaling, clipping and optimizer update in their original order.
"""

import inspect
import types

import torch

from turbo_physai.optimizations.models.openwam.zero_fused import BLOCK, upload_metadata


def selected_gradients(optimizer, gradients, params):
    from deepspeed.runtime.zero.stage_1_and_2 import is_model_parallel_parameter

    return [
        g.detach()
        for g, p in zip(gradients, params)
        if g is not None
        and not (
            getattr(p, "ds_pipe_replicated", False)
            or getattr(p, "pipe_replicated", False)
        )
        and (optimizer.model_parallel_rank == 0 or is_model_parallel_parameter(p))
    ]


def norm_rows(lengths):
    return tuple(
        (index, offset, min(BLOCK, count - offset))
        for index, count in enumerate(lengths)
        for offset in range(0, count, BLOCK)
    )


class NormOverlap:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.stream = None
        self.pending = {}
        self.plans = {}
        self.metadata_bytes = 0
        self.in_epilogue = False

    def clear(self):
        # Cover overflow, skipped consumption, repeated/reentrant epilogues and
        # zero_grad before a new writer can touch the same allocations.
        for _, _, _, event in self.pending.values():
            torch.cuda.current_stream(self.stream.device).wait_event(event)
        self.pending.clear()

    def compute(self, grads):
        if not grads or any(
            not g.is_cuda
            or not g.is_contiguous()
            or g.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or g.dtype != grads[0].dtype
            or g.device != grads[0].device
            or g.numel() >= 2**31
            for g in grads
        ):
            return None
        from turbo_physai.optimizations.models.openwam.zero_kernels import (
            norm_partials_kernel,
        )
        import triton.language as tl

        device = grads[0].device
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        producer = torch.cuda.current_stream(device)
        self.stream.wait_stream(producer)
        self.stream.wait_stream(self.optimizer.reduction_stream)
        lengths = tuple(g.numel() for g in grads)
        key = (device, lengths)
        with torch.cuda.stream(self.stream):
            if key not in self.plans:
                if len(self.plans) >= 64:
                    return None
                rows = norm_rows(lengths)
                if not rows or self.metadata_bytes + len(rows) * 12 > 16 * 1024**2:
                    return None
                self.plans[key] = (len(rows), upload_metadata(rows, device))
                self.metadata_bytes += len(rows) * 12
            tiles, metadata = self.plans[key]
            for grad in grads:
                grad.record_stream(self.stream)
            pointers = torch.tensor(
                [g.data_ptr() for g in grads], dtype=torch.int64, pin_memory=True
            ).to(device=device, non_blocking=True)
            partials = torch.empty(tiles, dtype=torch.float64, device=device)
            dtype = {
                torch.bfloat16: tl.bfloat16,
                torch.float16: tl.float16,
                torch.float32: tl.float32,
            }[grads[0].dtype]
            norm_partials_kernel[(tiles,)](
                pointers,
                metadata,
                partials,
                dtype,
                BLOCK=BLOCK,
                num_warps=4,
                enable_fp_fusion=False,
            )
            # Same local squared-norm -> float32 -> DP SUM order as zero.py.
            # Summation order differs slightly from foreach_norm's per-tensor
            # sqrt/square; compare clipping-edge and updated weights remotely.
            value = partials.sum().float()
            event = torch.cuda.Event()
            event.record(self.stream)
        return value, event

    def prepare(self, gradients, params):
        grads = selected_gradients(self.optimizer, gradients, params)
        result = self.compute(grads)
        if result is not None:
            value, event = result
            # Only IDs and scalar results survive the call; never retain a
            # gradient list/view. The owning averaged_gradients list is alive
            # until DeepSpeed consumes it; every epilogue starts a new generation.
            self.pending[id(gradients)] = (id(gradients), id(params), value, event)

    def take(self, gradients, params):
        entry = self.pending.get(id(gradients))
        if entry is None or entry[1] != id(params):
            return None
        _, _, value, event = self.pending.pop(id(gradients))
        current = torch.cuda.current_stream(value.device)
        current.wait_event(event)
        value.record_stream(current)
        return value

    def install(self):
        opt = self.optimizer
        original_flat = opt.get_flat_partition
        original_epilogue = opt.independent_gradient_partition_epilogue
        original_zero = opt.zero_grad
        signature = inspect.signature(original_flat)
        expected = (
            "tensor_list",
            "first_offset",
            "partition_size",
            "dtype",
            "device",
            "param_group_idx",
            "return_tensor_list",
        )
        if tuple(signature.parameters) != expected:
            raise RuntimeError(
                "Unsupported DeepSpeed get_flat_partition interface for norm overlap"
            )

        def flat(this, *args, **kwargs):
            result = original_flat(*args, **kwargs)
            if self.in_epilogue and this.is_gradient_accumulation_boundary:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                if bound.arguments["return_tensor_list"] and isinstance(result, list):
                    self.prepare(result, bound.arguments["tensor_list"])
            return result

        def epilogue(this):
            self.clear()
            self.in_epilogue = True
            try:
                return original_epilogue()
            except BaseException:
                self.clear()
                raise
            finally:
                self.in_epilogue = False

        def zero_grad(this, *args, **kwargs):
            self.clear()
            return original_zero(*args, **kwargs)

        opt.get_flat_partition = types.MethodType(flat, opt)
        opt.independent_gradient_partition_epilogue = types.MethodType(epilogue, opt)
        opt.zero_grad = types.MethodType(zero_grad, opt)
