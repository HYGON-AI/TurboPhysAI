# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""Remove per-step host synchronizations from the training loop.

The default loop reads loss/grad-norm with .item() every step, DeepSpeed's
throughput timer calls cuda.synchronize() inside engine.step, and a few small
CPU tensors are uploaded with blocking copies. Each one drains the GPU queue,
so the host must relaunch the next work from an idle device. Here metrics are
reduced on the device, copied to pinned memory without blocking and reported
one step late; small uploads use pinned non-blocking copies.
"""

import torch

from turbo_physai.optimizations.models.openwam import enabled


def active():
    return enabled("host_async")


def configure_deepspeed(config):
    # The throughput timer is only used for DeepSpeed's own steps_per_print
    # report; when enabled it synchronizes the device at every engine.step.
    timers = dict(config.get("timers") or {})
    throughput = dict(timers.get("throughput") or {})
    throughput.update(enabled=False, synchronized=False)
    timers["throughput"] = throughput
    config["timers"] = timers


def upload(tensor, device):
    """Pinned non-blocking H2D for small host tensors; others keep .to()."""
    target = torch.device(device)
    if tensor.device.type == "cpu" and target.type == "cuda":
        return tensor.pin_memory().to(device=target, non_blocking=True)
    return tensor.to(device=target)


def device_scalar(value, device):
    """Detach device values or fill constants without a blocking scalar H2D."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.full((), float(value), device=device, dtype=torch.float32)


class LaggedMetrics:
    """Same cross-rank mean as reduce_step_metrics, delivered one step later."""

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.pending = None
        self.stream = None

    def push(self, losses, grad_norm, info):
        from turbo_physai.optimizations.models.openwam import deferred_gather

        device = losses["total"].device
        values = torch.stack(
            [
                device_scalar(value, device).reshape(())
                for value in (
                    losses["total"],
                    losses["video"],
                    losses["action"],
                    grad_norm,
                )
            ]
        )
        host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
        event = torch.cuda.Event()
        if (
            deferred_gather.PENDING is not None
            and self.accelerator is not None
            and self.accelerator.num_processes > 1
        ):
            self._gather_aside(values, host, event)
        else:
            if self.accelerator is not None and self.accelerator.num_processes > 1:
                values = self.accelerator.gather(values.reshape(1, -1)).mean(dim=0)
            host.copy_(values, non_blocking=True)
            event.record()
        ready = self.flush()
        self.pending = (event, host, info)
        return ready

    def _gather_aside(self, values, host, event):
        """Gather on a side stream while deferred-gather broadcasts are in flight.

        Ranks are no longer aligned by a step-end AllGather, so this small
        collective can take as long as the rank skew; the training stream would
        otherwise wait for it before the next forward. Same values and mean as
        accelerator.gather over the default group.
        """
        import torch.distributed as dist

        if self.stream is None:
            self.stream = torch.cuda.Stream(device=values.device)
        gathered = torch.empty(
            (dist.get_world_size(), values.numel()),
            dtype=values.dtype,
            device=values.device,
        )
        work = dist.all_gather_into_tensor(gathered, values, async_op=True)
        with torch.cuda.stream(self.stream):
            work.wait()
            host.copy_(gathered.mean(dim=0), non_blocking=True)
            event.record(self.stream)
        gathered.record_stream(self.stream)

    def flush(self):
        if self.pending is None:
            return []
        event, host, info = self.pending
        self.pending = None
        event.synchronize()
        loss, video, action, grad_norm = host.tolist()
        return [
            (
                info,
                {
                    "loss_total": loss,
                    "loss_video": video,
                    "loss_action": action,
                    "grad_norm": grad_norm,
                },
            )
        ]


def rope_freqs(dit, grid, device, build):
    """Device-resident RoPE table per latent grid; the CPU table is immutable."""
    if not active() or torch.compiler.is_compiling():
        return build().to(device)
    cache = dit.__dict__.setdefault("_openwam_rope_cache", {})
    key = (tuple(int(size) for size in grid), str(device))
    value = cache.get(key)
    if value is None:
        value = cache[key] = upload(build(), device)
    return value
